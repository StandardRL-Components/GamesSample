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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_TRACK = (80, 80, 90)
        self.COLOR_RUMBLE = (200, 40, 40)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_OPPONENT = (255, 120, 0)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_BOOST = (100, 255, 100)
        self.COLOR_CHECKPOINT = (255, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_SMOKE = (200, 200, 200)

        # Game constants
        self.MAX_STEPS = 3000
        self.NUM_LAPS = 2

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_state = {}
        self.opponent_state = {}
        self.track = []
        self.checkpoints = []
        self.boost_pads = []
        self.projectiles = []
        self.particles = []

        # self.reset() is called in validate_implementation to ensure a valid state
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_track()

        start_angle = math.atan2(
            self.track[1][1] - self.track[0][1],
            self.track[1][0] - self.track[0][0]
        )

        self.player_state = {
            "pos": np.array(self.track[0], dtype=np.float64) + np.array([0, -15]),
            "vel": np.array([0.0, 0.0]),
            "angle": start_angle,
            "drift_counter": 0,
            "on_track": True,
            "weapon_cooldown": 0,
            "lap": 0,
            "next_checkpoint": 0,
            "dist_to_next_checkpoint": 0.0,  # Placeholder, will be set below
        }

        # Set the initial distance correctly after player_state is defined
        self.player_state["dist_to_next_checkpoint"] = self._get_dist_to_next_checkpoint()

        self.opponent_state = {
            "pos": np.array(self.track[0], dtype=np.float64) + np.array([0, 15]),
            "vel": np.array([0.0, 0.0]),
            "angle": start_angle,
            "target_node": 1,
            "health": 100,
        }

        self.projectiles = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        reward = 0
        self.steps += 1

        prev_dist = self.player_state["dist_to_next_checkpoint"]

        self._update_player(movement, space_held, shift_held)
        self._update_opponent()
        self._update_projectiles()
        self._update_particles()

        reward += self._handle_collisions_and_checkpoints()

        # Reward for progress
        current_dist = self._get_dist_to_next_checkpoint()
        reward += (prev_dist - current_dist) * 0.1
        self.player_state["dist_to_next_checkpoint"] = current_dist

        if not self.player_state["on_track"]:
            reward -= 0.1  # Penalty for being off-track

        self.score += reward
        terminated = self._check_termination()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_track(self):
        self.track = []
        self.checkpoints = []
        self.boost_pads = []

        center_x, center_y = self.width / 2, self.height / 2
        num_points = 10
        radius_x, radius_y = self.width * 0.4, self.height * 0.35

        points = []
        for i in range(num_points):
            angle = (2 * math.pi / num_points) * i
            rx = radius_x * (1 + self.np_random.uniform(-0.2, 0.2))
            ry = radius_y * (1 + self.np_random.uniform(-0.2, 0.2))
            x = center_x + rx * math.cos(angle)
            y = center_y + ry * math.sin(angle)
            points.append((x, y))

        # Create a smooth path using Catmull-Rom splines
        self.track = []
        for i in range(num_points):
            p0 = points[(i - 1 + num_points) % num_points]
            p1 = points[i]
            p2 = points[(i + 1) % num_points]
            p3 = points[(i + 2) % num_points]
            for t_int in range(10):
                t = t_int / 10.0
                x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t ** 2 + (
                            -p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t ** 3)
                y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t ** 2 + (
                            -p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t ** 3)
                if not self.track or np.linalg.norm(np.array([x, y]) - self.track[-1]) > 5:
                    self.track.append(np.array([x, y]))

        # Create checkpoints and boost pads
        num_checkpoints = 5
        if len(self.track) > num_checkpoints:
            for i in range(num_checkpoints):
                idx = (i * (len(self.track) // num_checkpoints)) % len(self.track)
                if idx > 0:
                    self.checkpoints.append(self.track[idx])
        if not self.checkpoints: # Ensure at least one checkpoint
            self.checkpoints.append(self.track[len(self.track)//2])


        for i, p in enumerate(self.track):
            if i % (len(self.track) // 7) == 20:  # Randomly place some boost pads
                self.boost_pads.append(p)

    def _update_player(self, movement, space_held, shift_held):
        state = self.player_state

        # Input handling
        turn_speed = 0.08
        accel = 0.3
        brake = 0.5
        max_speed = 8.0

        is_drifting = shift_held and np.linalg.norm(state["vel"]) > 2.0

        if movement == 1:  # Up
            force = np.array([math.cos(state["angle"]), math.sin(state["angle"])]) * accel
            state["vel"] += force
        if movement == 2:  # Down
            state["vel"] *= (1.0 - brake / 10.0)
        if movement == 3:  # Left
            state["angle"] -= turn_speed * (1 + 0.5 * is_drifting)
        if movement == 4:  # Right
            state["angle"] += turn_speed * (1 + 0.5 * is_drifting)

        # Physics update
        speed = np.linalg.norm(state["vel"])
        if speed > max_speed:
            state["vel"] = state["vel"] / speed * max_speed

        # Friction
        if is_drifting:
            state["drift_counter"] += 1
            if state["drift_counter"] % 3 == 0:
                # sound: tire screech
                self._create_particle(state["pos"], count=1, p_type='smoke', life=30, size=5)

            vel_dir = state["vel"] / (speed + 1e-6)
            forward_vel = np.dot(state["vel"], np.array([math.cos(state["angle"]), math.sin(state["angle"])]))
            right_vec = np.array([math.cos(state["angle"] + math.pi / 2), math.sin(state["angle"] + math.pi / 2)])
            lateral_vel = np.dot(state["vel"], right_vec)

            state["vel"] -= right_vec * lateral_vel * 0.1  # Reduced lateral friction
            state["vel"] *= 0.99
        else:
            state["drift_counter"] = 0
            state["vel"] *= 0.97  # Normal friction

        state["pos"] += state["vel"]

        # Weapon
        if state["weapon_cooldown"] > 0:
            state["weapon_cooldown"] -= 1
        if space_held and state["weapon_cooldown"] == 0:
            # sound: laser fire
            self._fire_projectile(state["pos"], state["angle"], self.COLOR_PROJECTILE, is_player=True)
            state["weapon_cooldown"] = 15
            self._create_particle(state["pos"] + np.array([math.cos(state["angle"]), math.sin(state["angle"])]) * 20,
                                  count=5, p_type='muzzle')

    def _update_opponent(self):
        state = self.opponent_state
        if state["health"] <= 0: return

        target_pos = self.track[state["target_node"]]
        direction_vec = target_pos - state["pos"]
        dist = np.linalg.norm(direction_vec)

        if dist < 40:
            state["target_node"] = (state["target_node"] + 1) % len(self.track)

        target_angle = math.atan2(direction_vec[1], direction_vec[0])
        angle_diff = (target_angle - state["angle"] + math.pi) % (2 * math.pi) - math.pi

        state["angle"] += np.clip(angle_diff, -0.07, 0.07)

        force = np.array([math.cos(state["angle"]), math.sin(state["angle"])]) * 0.2
        state["vel"] += force

        speed = np.linalg.norm(state["vel"])
        if speed > 5.0:
            state["vel"] = state["vel"] / speed * 5.0

        state["vel"] *= 0.98
        state["pos"] += state["vel"]

    def _update_projectiles(self):
        self.projectiles = [p for p in self.projectiles if
                            0 < p['pos'][0] < self.width and 0 < p['pos'][1] < self.height and p['life'] > 0]
        for p in self.projectiles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] -= p['decay']

    def _handle_collisions_and_checkpoints(self):
        reward = 0

        # Player on track / boost pads
        self.player_state["on_track"], _ = self._is_on_track(self.player_state["pos"])
        if not self.player_state["on_track"]:
            self.player_state["vel"] *= 0.9  # Off-track slow down

        for pad_pos in self.boost_pads:
            if np.linalg.norm(self.player_state["pos"] - pad_pos) < 20:
                # sound: boost pickup
                self.player_state["vel"] *= 1.5
                self._create_particle(self.player_state["pos"], count=20, p_type='boost', life=40, speed=3)
                reward += 5
                break

        # Checkpoints
        cp_pos = self.checkpoints[self.player_state["next_checkpoint"]]
        if np.linalg.norm(self.player_state["pos"] - cp_pos) < 30:
            # sound: checkpoint
            self.player_state["next_checkpoint"] += 1
            reward += 20
            if self.player_state["next_checkpoint"] >= len(self.checkpoints):
                self.player_state["next_checkpoint"] = 0
                self.player_state["lap"] += 1
                reward += 100 * (self.NUM_LAPS - self.player_state["lap"] + 1)

        # Projectile collisions
        for p in self.projectiles[:]:
            if p['is_player'] and self.opponent_state["health"] > 0:
                if np.linalg.norm(p['pos'] - self.opponent_state['pos']) < 15:
                    # sound: explosion
                    self.opponent_state['health'] -= 25
                    self.projectiles.remove(p)
                    self._create_particle(p['pos'], count=15, p_type='hit')
                    reward += 15
                    if self.opponent_state['health'] <= 0:
                        reward += 50
                        self._create_particle(self.opponent_state['pos'], count=50, p_type='explosion')

        return reward

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self.player_state["lap"] >= self.NUM_LAPS:
            self.score += 200  # Victory bonus
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_track()
        self._render_particles()
        if self.opponent_state["health"] > 0:
            self._render_car(self.opponent_state, self.COLOR_OPPONENT)
        else:  # Render wreckage
            pygame.draw.circle(self.screen, (50, 50, 50), self.opponent_state["pos"], 15)
        self._render_car(self.player_state, self.COLOR_PLAYER, has_glow=True)
        self._render_projectiles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self):
        track_width = 40
        rumble_width = 5

        if len(self.track) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, self.track, track_width)

            for i in range(len(self.track) - 1):
                p1 = self.track[i]
                p2 = self.track[i + 1]
                
                angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + math.pi / 2
                dx, dy = math.cos(angle), math.sin(angle)

                r1_start = p1 + np.array([dx, dy]) * (track_width / 2)
                r1_end = p2 + np.array([dx, dy]) * (track_width / 2)
                pygame.draw.line(self.screen, self.COLOR_RUMBLE, r1_start, r1_end, rumble_width)

                r2_start = p1 - np.array([dx, dy]) * (track_width / 2)
                r2_end = p2 - np.array([dx, dy]) * (track_width / 2)
                pygame.draw.line(self.screen, self.COLOR_RUMBLE, r2_start, r2_end, rumble_width)

        for pad_pos in self.boost_pads:
            pygame.gfxdraw.filled_circle(self.screen, int(pad_pos[0]), int(pad_pos[1]), 15, self.COLOR_BOOST + (100,))

        # Checkpoint line
        cp_idx = self.player_state["next_checkpoint"]
        cp_pos = self.checkpoints[cp_idx]
        if cp_idx > 0:
            prev_cp_pos = self.checkpoints[cp_idx - 1]
        else:
            prev_cp_pos = self.checkpoints[-1]

        angle = math.atan2(cp_pos[1] - prev_cp_pos[1], cp_pos[0] - prev_cp_pos[0]) + math.pi / 2
        dx, dy = math.cos(angle), math.sin(angle)

        p1 = cp_pos + np.array([dx, dy]) * 30
        p2 = cp_pos - np.array([dx, dy]) * 30
        pygame.draw.line(self.screen, self.COLOR_CHECKPOINT, p1, p2, 5)

    def _render_car(self, state, color, has_glow=False):
        pos = state["pos"]
        angle = state["angle"]

        car_width, car_length = 12, 22

        points = [
            (-car_length / 2, -car_width / 2), (car_length / 2, -car_width / 2),
            (car_length / 2, car_width / 2), (-car_length / 2, car_width / 2)
        ]

        rotated_points = []
        for x, y in points:
            new_x = x * math.cos(angle) - y * math.sin(angle) + pos[0]
            new_y = x * math.sin(angle) + y * math.cos(angle) + pos[1]
            rotated_points.append((new_x, new_y))

        if has_glow:
            glow_color = color + (50,)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], glow_color)
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], glow_color)

        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_points], color)

        # Windshield
        windshield_points = [
            (car_length / 4, -car_width / 2 + 2), (car_length / 2 - 2, -car_width / 2 + 2),
            (car_length / 2 - 2, car_width / 2 - 2), (car_length / 4, car_width / 2 - 2)
        ]
        rotated_windshield = []
        for x, y in windshield_points:
            new_x = x * math.cos(angle) - y * math.sin(angle) + pos[0]
            new_y = x * math.sin(angle) + y * math.cos(angle) + pos[1]
            rotated_windshield.append((new_x, new_y))
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in rotated_windshield], (20, 20, 20))

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, p['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, p['color'] + (100,))

    def _render_particles(self):
        for p in self.particles:
            if p['size'] > 0:
                color = p['color'] + (int(255 * (p['life'] / p['max_life'])),)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_small.render(f"Time: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_WHITE)
        self.screen.blit(steps_text, (self.width - steps_text.get_width() - 10, 10))

        lap_text = self.font_main.render(f"Lap: {self.player_state['lap'] + 1}/{self.NUM_LAPS}", True, self.COLOR_WHITE)
        self.screen.blit(lap_text, (self.width / 2 - lap_text.get_width() / 2, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.player_state["lap"],
            "on_track": self.player_state["on_track"],
        }

    def _is_on_track(self, pos):
        min_dist = float('inf')
        closest_point = None
        for p in self.track:
            dist = np.linalg.norm(pos - p)
            if dist < min_dist:
                min_dist = dist
                closest_point = p
        return min_dist < 23, closest_point

    def _get_dist_to_next_checkpoint(self):
        return np.linalg.norm(self.player_state['pos'] - self.checkpoints[self.player_state['next_checkpoint']])

    def _fire_projectile(self, pos, angle, color, is_player):
        vel = np.array([math.cos(angle), math.sin(angle)]) * 12
        self.projectiles.append({
            'pos': pos + vel * 2, 'vel': vel, 'color': color,
            'life': 50, 'is_player': is_player
        })

    def _create_particle(self, pos, count, p_type, life=20, size=3, speed=2):
        for _ in range(count):
            if p_type == 'smoke':
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(0.1, 0.5)
                color = self.COLOR_SMOKE
                decay = 0.1
            elif p_type == 'boost':
                angle = self.player_state['angle'] + math.pi + self.np_random.uniform(-0.3, 0.3)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                color = self.COLOR_BOOST
                decay = 0.2
            elif p_type == 'muzzle':
                angle = self.player_state['angle'] + self.np_random.uniform(-0.5, 0.5)
                vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(1, 3)
                color = (255, 200, 50)
                life = 5
                decay = 0.5
            elif p_type == 'hit':
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(1, 4)
                color = self.COLOR_OPPONENT
                decay = 0.2
            elif p_type == 'explosion':
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(0.5, 5)
                color = self.np_random.choice([self.COLOR_OPPONENT, (255, 255, 0), (255, 50, 50)],
                                              p=[0.5, 0.25, 0.25])
                life = 40
                decay = 0.1
            else: # default
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(0.1, 0.5)
                color = self.COLOR_WHITE
                decay = 0.1


            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life,
                'size': size, 'decay': decay, 'color': color
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This is not a part of the official API, but a helper for development
        print("Validating implementation...")
        
        # 1. Reset the environment
        obs, info = self.reset()

        # 2. Check spaces
        assert self.action_space.contains(self.action_space.sample())
        assert self.observation_space.contains(obs)
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        assert obs.shape == (self.height, self.width, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # 3. Step the environment
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        
        # 4. Check step outputs
        assert self.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in the headless verification environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    print(env.user_guide)

    while not terminated:
        # Construct the action from keyboard state
        action = [0, 0, 0]  # [movement, space, shift]

        keys = pygame.key.get_pressed()

        # Movement (only one direction at a time)
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        if keys[pygame.K_SPACE]:
            action[1] = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)  # Run at 30 FPS

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # obs, info = env.reset() # Uncomment to play again
            # terminated = False

    env.close()