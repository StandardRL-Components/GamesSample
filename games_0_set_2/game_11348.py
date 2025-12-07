import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A turn-based physics game where you launch satellites to knock your opponent's pieces off the board. "
        "Unlock new satellite types and board layouts as you win."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to adjust power and ↑↓ to adjust angle. "
        "Press space to launch your satellite. Press shift to cycle through available ammo types."
    )
    auto_advance = False

    # --- Persistent Progression State ---
    # These exist for the lifetime of the Python process, as per the brief.
    board_wins = 0
    unlocked_satellite_types = [
        {"name": "Standard", "mass": 10, "radius": 12, "restitution": 0.9},
    ]

    SATELLITE_UNLOCK_SCHEDULE = {
        3: {"name": "Heavy", "mass": 20, "radius": 14, "restitution": 0.7},
        6: {"name": "Bouncy", "mass": 8, "radius": 10, "restitution": 1.0},
    }

    layout_level = 0
    LAYOUT_UNLOCK_SCHEDULE = {5, 10, 15}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.STALEMATE_THRESHOLD = 50  # Turns without a knockout

        # --- Colors ---
        self.COLOR_BG = (15, 19, 41)
        self.COLOR_BOARD = (50, 60, 90)
        self.COLOR_BOARD_GLOW = (70, 80, 120)
        self.COLOR_P1 = (0, 150, 255)
        self.COLOR_P2 = (255, 50, 50)
        self.COLOR_P1_GLOW = (0, 150, 255, 50)
        self.COLOR_P2_GLOW = (255, 50, 50, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TRAJECTORY = (255, 255, 255, 150)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_combo = pygame.font.SysFont("Consolas", 20, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.satellites = []
        self.particles = []
        self.combo_popups = []

        self.phase = "AWAITING_INPUT"  # "AWAITING_INPUT" or "SIMULATING_PHYSICS"
        self.current_player = 1

        self.launch_angle = 90.0
        self.launch_power = 50.0
        self.selected_satellite_idx = 0

        self.previous_space_held = False
        self.previous_shift_held = False

        self.turns_since_knockoff = 0
        self.last_launched_satellite = None
        self.turn_hit_opponent = False

        self.board_center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2 + 20)
        self.board_radius = 160
        self.launch_pad_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 20)

        # Initialize state variables
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.satellites = []
        self.particles = []
        self.combo_popups = []

        self.phase = "AWAITING_INPUT"
        self.current_player = 1

        self.launch_angle = 90.0
        self.launch_power = 50.0
        self.selected_satellite_idx = 0

        self.previous_space_held = False
        self.previous_shift_held = False

        self.turns_since_knockoff = 0
        self.last_launched_satellite = None
        self.turn_hit_opponent = False

        # Check for progression unlocks
        if self.board_wins in self.SATELLITE_UNLOCK_SCHEDULE and self.SATELLITE_UNLOCK_SCHEDULE[
            self.board_wins] not in self.unlocked_satellite_types:
            self.unlocked_satellite_types.append(self.SATELLITE_UNLOCK_SCHEDULE[self.board_wins])
        if self.board_wins in self.LAYOUT_UNLOCK_SCHEDULE:
            GameEnv.layout_level = list(self.LAYOUT_UNLOCK_SCHEDULE).index(self.board_wins) + 1

        self._setup_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        space_pressed = space_held and not self.previous_space_held
        shift_pressed = shift_held and not self.previous_shift_held

        if self.phase == "AWAITING_INPUT":
            # --- Handle Aiming ---
            if movement == 1: self.launch_angle = min(180.0, self.launch_angle + 1.5)  # Up
            elif movement == 2: self.launch_angle = max(0.0, self.launch_angle - 1.5)  # Down
            elif movement == 3: self.launch_power = max(10.0, self.launch_power - 1.0)  # Left
            elif movement == 4: self.launch_power = min(100.0, self.launch_power + 1.0)  # Right

            # --- Handle Satellite Selection ---
            if shift_pressed:
                self.selected_satellite_idx = (self.selected_satellite_idx + 1) % len(self.unlocked_satellite_types)
                # SFX: select_switch

            # --- Handle Launch ---
            if space_pressed:
                self.phase = "SIMULATING_PHYSICS"
                self._launch_satellite()
                reward -= 0.05  # Penalty for taking a shot (missing)
                self.turn_hit_opponent = False
                self.turns_since_knockoff += 1
                # SFX: launch

        elif self.phase == "SIMULATING_PHYSICS":
            physics_reward, sim_ended = self._run_physics_step()
            reward += physics_reward

            if sim_ended:
                if self.turn_hit_opponent:
                    reward += 0.15  # Net reward for a hit: -0.05 + 0.15 = +0.1

                # Check for win/loss
                p1_sats = sum(1 for s in self.satellites if s['player'] == 1)
                p2_sats = sum(1 for s in self.satellites if s['player'] == 2)

                if p1_sats == 0:
                    reward -= 100
                    terminated = True
                    # SFX: game_loss
                elif p2_sats == 0:
                    reward += 100
                    terminated = True
                    GameEnv.board_wins += 1  # Persist win
                    # SFX: game_win

                # Stalemate breaker
                if not terminated and self.turns_since_knockoff >= self.STALEMATE_THRESHOLD:
                    self._break_stalemate()
                    self.turns_since_knockoff = 0

                # Switch turn if game is not over
                if not terminated:
                    self.current_player = 3 - self.current_player  # Switch 1->2, 2->1
                    self.phase = "AWAITING_INPUT"
                    self.launch_angle = 90.0
                    self.launch_power = 50.0

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward  # Internal score tracking for display

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _launch_satellite(self):
        sat_type = self.unlocked_satellite_types[self.selected_satellite_idx]
        angle_rad = math.radians(self.launch_angle)
        launch_vel = pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * (self.launch_power / 5.0)

        new_sat = {
            "pos": self.launch_pad_pos.copy(),
            "vel": launch_vel,
            "player": self.current_player,
            "type": sat_type,
            "id": self.np_random.integers(1000, 10000)
        }
        self.satellites.append(new_sat)
        self.last_launched_satellite = new_sat

    def _run_physics_step(self):
        reward = 0.0
        sub_steps = 4  # More sub-steps for stability

        for _ in range(sub_steps):
            # Move satellites
            for sat in self.satellites:
                sat['pos'] += sat['vel'] / sub_steps
                sat['vel'] *= 0.999  # Damping/friction

            # Handle collisions
            for i in range(len(self.satellites)):
                for j in range(i + 1, len(self.satellites)):
                    sat1 = self.satellites[i]
                    sat2 = self.satellites[j]

                    dist_vec = sat1['pos'] - sat2['pos']
                    dist = dist_vec.length()
                    min_dist = sat1['type']['radius'] + sat2['type']['radius']

                    if dist < min_dist:
                        # Collision occurred
                        self._resolve_collision(sat1, sat2, dist_vec, dist, min_dist)
                        self._spawn_particles((sat1['pos'] + sat2['pos']) / 2, 20)
                        # SFX: satellite_collision

                        # Check if the just-launched satellite hit an opponent
                        if not self.turn_hit_opponent:
                            if (self.last_launched_satellite and sat1['id'] == self.last_launched_satellite['id'] and sat2['player'] != self.current_player) or \
                               (self.last_launched_satellite and sat2['id'] == self.last_launched_satellite['id'] and sat1['player'] != self.current_player):
                                self.turn_hit_opponent = True

        # Check for out of bounds and calculate knock-off rewards
        sats_to_remove = []
        combo = 0
        for sat in self.satellites:
            if sat['pos'].distance_to(self.board_center) > self.board_radius:
                sats_to_remove.append(sat)
                if sat['player'] != self.current_player:
                    combo += 1
                    reward += 1.0 + (0.5 * combo)
                    self.turns_since_knockoff = 0
                    self.combo_popups.append({"text": f"COMBO x{combo}!", "pos": sat['pos'].copy(), "life": 60})
                    # SFX: point_scored
                else:  # Player knocked their own satellite off
                    reward -= 1.0
                    # SFX: point_lost

        if sats_to_remove:
            self.satellites = [s for s in self.satellites if s not in sats_to_remove]

        # Update particles and combo popups
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        self.combo_popups = [c for c in self.combo_popups if c['life'] > 0]
        for c in self.combo_popups:
            c['life'] -= 1

        # Check if simulation is over
        sim_ended = all(s['vel'].length_squared() < 0.01 for s in self.satellites)

        return reward, sim_ended

    def _resolve_collision(self, sat1, sat2, dist_vec, dist, min_dist):
        # Prevent overlap
        overlap = min_dist - dist
        if dist == 0: dist_vec = pygame.Vector2(1, 0); dist = 1
        push_vec = dist_vec.normalize() * overlap
        sat1['pos'] += push_vec / 2
        sat2['pos'] -= push_vec / 2

        # Collision response
        normal = dist_vec.normalize()
        tangent = pygame.Vector2(-normal.y, normal.x)

        m1, m2 = sat1['type']['mass'], sat2['type']['mass']
        restitution = min(sat1['type']['restitution'], sat2['type']['restitution'])

        v1_n = sat1['vel'].dot(normal)
        v1_t = sat1['vel'].dot(tangent)
        v2_n = sat2['vel'].dot(normal)
        v2_t = sat2['vel'].dot(tangent)

        v1_n_new = (v1_n * (m1 - m2 * restitution) + 2 * m2 * v2_n) / (m1 + m2)
        v2_n_new = (v2_n * (m2 - m1 * restitution) + 2 * m1 * v1_n) / (m1 + m2)

        sat1['vel'] = normal * v1_n_new + tangent * v1_t
        sat2['vel'] = normal * v2_n_new + tangent * v2_t

    def _setup_board(self):
        num_sats_per_player = 4
        # Place player 1 satellites
        for i in range(num_sats_per_player):
            angle = math.pi / 2 + (i - (num_sats_per_player - 1) / 2) * 0.4
            pos = self.board_center + pygame.Vector2(math.cos(angle), math.sin(angle)) * (self.board_radius * 0.7)
            self.satellites.append({"pos": pos, "vel": pygame.Vector2(0, 0), "player": 1,
                                    "type": self.unlocked_satellite_types[0], "id": self.np_random.integers(0, 1000)})

        # Place player 2 satellites
        for i in range(num_sats_per_player):
            angle = -math.pi / 2 + (i - (num_sats_per_player - 1) / 2) * 0.4
            pos = self.board_center + pygame.Vector2(math.cos(angle), math.sin(angle)) * (self.board_radius * 0.7)
            self.satellites.append({"pos": pos, "vel": pygame.Vector2(0, 0), "player": 2,
                                    "type": self.unlocked_satellite_types[0], "id": self.np_random.integers(0, 1000)})

        # Add obstacles based on layout level
        if GameEnv.layout_level > 0:
            num_obstacles = GameEnv.layout_level * 2
            for i in range(num_obstacles):
                angle = (2 * math.pi / num_obstacles) * i
                radius = self.board_radius * (0.2 + 0.1 * (i % 2))
                pos = self.board_center + pygame.Vector2(math.cos(angle), math.sin(angle)) * radius
                self.satellites.append({
                    "pos": pos, "vel": pygame.Vector2(0, 0), "player": 0,  # Neutral
                    "type": {"name": "Obstacle", "mass": 50, "radius": 8, "restitution": 0.5},
                    "id": self.np_random.integers(0, 1000)
                })

    def _break_stalemate(self):
        for sat in self.satellites:
            force = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * 0.5
            sat['vel'] += force

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for _ in range(50):
            x = (self.np_random.integers(0, self.WIDTH * 2) + self.steps) % self.WIDTH
            y = self.np_random.integers(0, self.HEIGHT)
            pygame.draw.circle(self.screen, (100, 100, 120), (x, y), self.np_random.choice([0, 1]))

        # --- Game Board ---
        pygame.gfxdraw.filled_circle(self.screen, int(self.board_center.x), int(self.board_center.y),
                                     self.board_radius, self.COLOR_BOARD)
        pygame.gfxdraw.aacircle(self.screen, int(self.board_center.x), int(self.board_center.y), self.board_radius,
                                self.COLOR_BOARD_GLOW)

        # --- Render Game Elements ---
        self._render_game_elements()

        # --- Render UI ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_elements(self):
        # --- Launch Pad & Trajectory ---
        if self.phase == "AWAITING_INPUT":
            pad_color = self.COLOR_P1 if self.current_player == 1 else self.COLOR_P2
            pygame.draw.circle(self.screen, pad_color, (int(self.launch_pad_pos.x), int(self.launch_pad_pos.y)), 10)
            pygame.gfxdraw.aacircle(self.screen, int(self.launch_pad_pos.x), int(self.launch_pad_pos.y), 10, pad_color)

            angle_rad = math.radians(self.launch_angle)
            end_pos = self.launch_pad_pos + pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * self.launch_power
            self._draw_dashed_line(self.launch_pad_pos, end_pos, color=self.COLOR_TRAJECTORY)

        # --- Satellites ---
        for sat in self.satellites:
            pos = (int(sat['pos'].x), int(sat['pos'].y))
            radius = sat['type']['radius']
            glow_radius = int(radius * 1.5)

            if sat['player'] == 1:
                color, glow_color = self.COLOR_P1, self.COLOR_P1_GLOW
            elif sat['player'] == 2:
                color, glow_color = self.COLOR_P2, self.COLOR_P2_GLOW
            else:
                color, glow_color = (100, 100, 100), (120, 120, 120, 50)

            # Glow effect
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255, 255, 255))

        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (255, 200, 50, alpha)
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (2, 2), 2)
            self.screen.blit(s, (int(p['pos'].x - 2), int(p['pos'].y - 2)))

        # --- Combo Popups ---
        for c in self.combo_popups:
            alpha = int(255 * (c['life'] / 60))
            text_surf = self.font_combo.render(c['text'], True, self.COLOR_TEXT)
            text_surf.set_alpha(alpha)
            pos = (int(c['pos'].x - text_surf.get_width() / 2),
                   int(c['pos'].y - text_surf.get_height() / 2 - (60 - c['life']) / 2))
            self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # Player scores/remaining satellites
        p1_sats = sum(1 for s in self.satellites if s['player'] == 1)
        p2_sats = sum(1 for s in self.satellites if s['player'] == 2)

        p1_text = self.font_large.render(f"P1: {p1_sats}", True, self.COLOR_P1)
        p2_text = self.font_large.render(f"P2: {p2_sats}", True, self.COLOR_P2)
        self.screen.blit(p1_text, (20, 10))
        self.screen.blit(p2_text, (self.WIDTH - p2_text.get_width() - 20, 10))

        # Turn indicator and selected satellite
        if self.phase == "AWAITING_INPUT":
            turn_color = self.COLOR_P1 if self.current_player == 1 else self.COLOR_P2
            turn_text_str = f"PLAYER {self.current_player} TURN"
            turn_text = self.font_large.render(turn_text_str, True, turn_color)
            self.screen.blit(turn_text, (self.WIDTH / 2 - turn_text.get_width() / 2, self.HEIGHT - 70))

            sat_type = self.unlocked_satellite_types[self.selected_satellite_idx]
            sat_text_str = f"AMMO: {sat_type['name']} (SHIFT to change)"
            sat_text = self.font_small.render(sat_text_str, True, self.COLOR_TEXT)
            self.screen.blit(sat_text, (self.WIDTH / 2 - sat_text.get_width() / 2, self.HEIGHT - 40))
        elif self.phase == "SIMULATING_PHYSICS":
            sim_text = self.font_large.render("SIMULATING...", True, self.COLOR_TEXT)
            self.screen.blit(sim_text, (self.WIDTH / 2 - sim_text.get_width() / 2, self.HEIGHT - 70))

    def _get_info(self):
        p1_sats = sum(1 for s in self.satellites if s['player'] == 1)
        p2_sats = sum(1 for s in self.satellites if s['player'] == 2)
        return {
            "score": self.score,
            "steps": self.steps,
            "p1_satellites": p1_sats,
            "p2_satellites": p2_sats,
            "current_player": self.current_player,
            "phase": self.phase,
            "board_wins": self.board_wins,
        }

    def _spawn_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 31)
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": life, "max_life": life})

    def _draw_dashed_line(self, start_pos, end_pos, color, dash_length=5, gap_length=3):
        v = end_pos - start_pos
        length = v.length()
        if length == 0: return
        direction = v.normalize()

        current_pos = start_pos.copy()
        traveled = 0
        while traveled < length:
            draw_end = current_pos + direction * dash_length
            if traveled + dash_length > length:
                draw_end = end_pos

            # Create a surface for alpha transparency
            line_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.aaline(line_surf, color, current_pos, draw_end)
            self.screen.blit(line_surf, (0, 0))

            current_pos += direction * (dash_length + gap_length)
            traveled += dash_length + gap_length

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Satellite Takedown")
    clock = pygame.time.Clock()

    terminated = False

    # --- Manual Control Mapping ---
    # ARROWS: Aim angle and power
    # SPACE: Launch
    # SHIFT: Change satellite type

    while not terminated:
        movement = 0  # none
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Phase: {info['phase']}")

        if terminated:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']:.2f}")
            # Reset for a new game
            # obs, info = env.reset()
            # terminated = False

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()