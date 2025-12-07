import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:18:00.637304
# Source Brief: brief_02091.md
# Brief Index: 2091
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a physics-based arcade game.

    **Objective:** Control a bouncing ball to hit 10 moving targets within 60 seconds.
    **Challenge:** Gravity shifts its direction by 15 degrees with every successful hit,
                 making trajectory prediction increasingly difficult.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `actions[0]` (Movement): Adjusts launch angle while aiming.
        - 0: None
        - 1: Up (aim higher)
        - 2: Down (aim lower)
        - 3: Left (aim lower)
        - 4: Right (aim higher)
    - `actions[1]` (Space): Hold to charge power, release to launch the ball.
        - 0: Released
        - 1: Held
    - `actions[2]` (Shift): Hold to reset the ball to the launchpad.
        - 0: Released
        - 1: Held

    **Observation Space:** `Box(shape=(400, 640, 3), dtype=np.uint8)`
    - A 640x400 RGB image of the current game state.

    **Rewards:**
    - +1.0 for each target hit.
    - +10.0 for hitting all targets (winning).
    - -10.0 for running out of time (losing).
    - +0.1 for getting closer to the nearest target.
    - -0.01 per step (time penalty).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch a ball to hit all moving targets. Be careful, as the direction of gravity "
        "shifts with every successful hit, making each shot a new challenge."
    )
    user_guide = (
        "Use the ↑/↓ arrow keys to aim. Hold space to charge your shot and release to fire. "
        "Press shift to reset the ball."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds at 30 FPS

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_TARGET = (0, 150, 255)
    COLOR_TARGET_HIT = (255, 50, 100)
    COLOR_WALL = (100, 100, 120)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TRAJECTORY = (255, 255, 255, 50)
    COLOR_GRAVITY_ARROW = (255, 200, 0)
    COLOR_POWER_BAR_BG = (50, 50, 70)
    COLOR_POWER_BAR_FG = (255, 255, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 60)

        self.game_state = "AIMING"
        self.launch_pos = np.array([60.0, self.HEIGHT / 2])
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.ball_radius = 12
        self.launch_angle = -math.pi / 4
        self.launch_power = 0.0
        self.max_launch_power = 15.0
        self.gravity_angle_deg = 90.0
        self.gravity = np.zeros(2)
        self.targets = []
        self.num_targets_hit = 0
        self.total_targets = 10
        self.was_space_held = False
        self.particles = []
        self.last_dist_to_target = float('inf')
        self.steps = 0
        self.score = 0.0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.was_space_held = False
        self.num_targets_hit = 0
        self.gravity_angle_deg = 90.0
        self.particles = []
        self.targets = self._generate_targets()
        self._reset_ball()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Time penalty

        # --- Handle Input ---
        if shift_held:
            # // SFX: Reset sound
            self._reset_ball()
        elif self.game_state == "AIMING":
            self._handle_aiming_input(movement, space_held)
        
        # --- Update Game Logic ---
        if self.game_state == "IN_FLIGHT":
            self._update_ball_physics()
            hit_reward = self._check_collisions()
            reward += hit_reward
            dist_reward = self._calculate_distance_reward()
            reward += dist_reward
        
        self._update_targets()
        self._update_particles()
        
        self.was_space_held = space_held
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if self.num_targets_hit == self.total_targets:
            # // SFX: Win fanfare
            reward += 10.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # // SFX: Lose sound
            reward -= 10.0
            terminated = True
            self.game_over = True
            
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _reset_ball(self):
        self.game_state = "AIMING"
        self.ball_pos = self.launch_pos.copy()
        self.ball_vel = np.array([0.0, 0.0])
        self.launch_power = 0.0
        self._update_gravity_vector()
        self.last_dist_to_target = self._get_dist_to_nearest_target()

    def _generate_targets(self):
        targets = []
        padding = 50
        for _ in range(self.total_targets):
            center_x = self.np_random.uniform(self.WIDTH / 2, self.WIDTH - padding)
            center_y = self.np_random.uniform(padding, self.HEIGHT - padding)
            orbit_radius = self.np_random.uniform(15, 40)
            orbit_speed = self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1])
            start_angle = self.np_random.uniform(0, 2 * math.pi)
            
            targets.append({
                "center": np.array([center_x, center_y]),
                "radius": 10,
                "orbit_radius": orbit_radius,
                "orbit_speed": orbit_speed,
                "angle": start_angle,
                "pos": np.array([center_x + orbit_radius * math.cos(start_angle), 
                                 center_y + orbit_radius * math.sin(start_angle)]),
                "hit": False,
                "hit_timer": 0
            })
        return targets

    def _handle_aiming_input(self, movement, space_held):
        angle_step = 0.03  # Radians per step
        if movement == 1 or movement == 4:  # Up or Right
            self.launch_angle -= angle_step
        elif movement == 2 or movement == 3:  # Down or Left
            self.launch_angle += angle_step
        self.launch_angle = np.clip(self.launch_angle, -math.pi * 0.9, math.pi * 0.9)

        if space_held:
            self.launch_power = min(self.max_launch_power, self.launch_power + 0.25)
        
        if not space_held and self.was_space_held and self.launch_power > 1.0:
            # // SFX: Launch swoosh
            self.game_state = "IN_FLIGHT"
            vel_x = self.launch_power * math.cos(self.launch_angle)
            vel_y = self.launch_power * math.sin(self.launch_angle)
            self.ball_vel = np.array([vel_x, vel_y])

    def _update_ball_physics(self):
        self.ball_vel += self.gravity
        self.ball_pos += self.ball_vel

    def _check_collisions(self):
        hit_reward = 0.0
        bounce_dampening = 0.85
        
        # Wall collisions
        if self.ball_pos[0] - self.ball_radius < 0:
            self.ball_pos[0] = self.ball_radius
            self.ball_vel[0] *= -bounce_dampening
            # // SFX: Wall bounce
        elif self.ball_pos[0] + self.ball_radius > self.WIDTH:
            self.ball_pos[0] = self.WIDTH - self.ball_radius
            self.ball_vel[0] *= -bounce_dampening
            # // SFX: Wall bounce
        if self.ball_pos[1] - self.ball_radius < 0:
            self.ball_pos[1] = self.ball_radius
            self.ball_vel[1] *= -bounce_dampening
            # // SFX: Wall bounce
        elif self.ball_pos[1] + self.ball_radius > self.HEIGHT:
            self.ball_pos[1] = self.HEIGHT - self.ball_radius
            self.ball_vel[1] *= -bounce_dampening
            # // SFX: Wall bounce

        # Target collisions
        for target in self.targets:
            if not target["hit"]:
                dist = np.linalg.norm(self.ball_pos - target["pos"])
                if dist < self.ball_radius + target["radius"]:
                    # // SFX: Target hit explosion
                    target["hit"] = True
                    target["hit_timer"] = self.FPS  # Linger for 1 second
                    self.num_targets_hit += 1
                    hit_reward += 1.0
                    self.gravity_angle_deg = (self.gravity_angle_deg + 15) % 360
                    self._update_gravity_vector()
                    self._spawn_particles(target["pos"], self.COLOR_TARGET_HIT)
        return hit_reward

    def _update_targets(self):
        for target in self.targets:
            if not target["hit"]:
                target["angle"] += target["orbit_speed"]
                target["pos"][0] = target["center"][0] + target["orbit_radius"] * math.cos(target["angle"])
                target["pos"][1] = target["center"][1] + target["orbit_radius"] * math.sin(target["angle"])
            elif target["hit_timer"] > 0:
                target["hit_timer"] -= 1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 0.03

    def _spawn_particles(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': 1.0,
                'color': color
            })

    def _update_gravity_vector(self):
        gravity_magnitude = 0.15
        rad = math.radians(self.gravity_angle_deg)
        self.gravity = np.array([math.cos(rad) * gravity_magnitude, math.sin(rad) * gravity_magnitude])

    def _get_dist_to_nearest_target(self):
        unhit_targets = [t for t in self.targets if not t["hit"]]
        if not unhit_targets:
            return 0.0
        
        distances = [np.linalg.norm(self.ball_pos - t["pos"]) for t in unhit_targets]
        return min(distances) if distances else float('inf')

    def _calculate_distance_reward(self):
        if self.game_state != "IN_FLIGHT":
            return 0.0

        dist = self._get_dist_to_nearest_target()
        reward = (self.last_dist_to_target - dist) * 0.1
        self.last_dist_to_target = dist
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        # Targets
        for target in self.targets:
            if target["hit_timer"] > 0:
                self._render_glow(self.screen, self.COLOR_TARGET_HIT, target["pos"], target["radius"], 10)
                pygame.gfxdraw.filled_circle(self.screen, int(target["pos"][0]), int(target["pos"][1]), target["radius"], self.COLOR_TARGET_HIT)
            elif not target["hit"]:
                pulse = (1 + math.sin(self.steps * 0.1)) * 2
                self._render_glow(self.screen, self.COLOR_TARGET, target["pos"], target["radius"] + pulse, 15)
                pygame.gfxdraw.filled_circle(self.screen, int(target["pos"][0]), int(target["pos"][1]), target["radius"], self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, int(target["pos"][0]), int(target["pos"][1]), target["radius"], self.COLOR_TARGET)

        # Particles
        for p in self.particles:
            alpha = max(0, int(p['life'] * 255))
            color = (*p['color'], alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # Ball
        self._render_glow(self.screen, self.COLOR_PLAYER, self.ball_pos, self.ball_radius, 20)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.ball_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.ball_radius, self.COLOR_PLAYER)

        # Aiming assists
        if self.game_state == "AIMING":
            self._render_trajectory_prediction()
            self._render_power_bar()

    def _render_glow(self, surface, color, center, radius, intensity):
        for i in range(intensity, 0, -2):
            alpha = 100 - (i / intensity * 100)
            glow_color = (*color, int(alpha))
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius + i), glow_color)
    
    def _render_trajectory_prediction(self):
        sim_pos = self.ball_pos.copy()
        sim_vel = np.array([self.launch_power * math.cos(self.launch_angle), 
                            self.launch_power * math.sin(self.launch_angle)])
        
        for i in range(60): # Predict 2 seconds into the future
            sim_vel += self.gravity
            sim_pos += sim_vel
            if i % 4 == 0:
                pygame.gfxdraw.filled_circle(self.screen, int(sim_pos[0]), int(sim_pos[1]), 2, self.COLOR_TRAJECTORY)

    def _render_power_bar(self):
        bar_x, bar_y = int(self.launch_pos[0]) - self.ball_radius - 20, int(self.launch_pos[1]) - 50
        bar_width, bar_height = 10, 100
        power_ratio = self.launch_power / self.max_launch_power
        
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        fill_height = int(bar_height * power_ratio)
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FG, (bar_x, bar_y + bar_height - fill_height, bar_width, fill_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

    def _render_ui(self):
        # Targets Hit
        targets_text = self.font_ui.render(f"TARGETS: {self.num_targets_hit}/{self.total_targets}", True, self.COLOR_UI_TEXT)
        self.screen.blit(targets_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Gravity Arrow
        arrow_center = (self.WIDTH - 40, self.HEIGHT - 40)
        rad = math.radians(self.gravity_angle_deg)
        arrow_end = (arrow_center[0] + 25 * math.cos(rad), arrow_center[1] + 25 * math.sin(rad))
        pygame.draw.line(self.screen, self.COLOR_GRAVITY_ARROW, arrow_center, arrow_end, 3)
        # Arrowhead
        p1 = (arrow_end[0] - 8 * math.cos(rad - 0.4), arrow_end[1] - 8 * math.sin(rad - 0.4))
        p2 = (arrow_end[0] - 8 * math.cos(rad + 0.4), arrow_end[1] - 8 * math.sin(rad + 0.4))
        pygame.draw.line(self.screen, self.COLOR_GRAVITY_ARROW, arrow_end, p1, 3)
        pygame.draw.line(self.screen, self.COLOR_GRAVITY_ARROW, arrow_end, p2, 3)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        
        if self.num_targets_hit == self.total_targets:
            text = "MISSION COMPLETE"
            color = self.COLOR_PLAYER
        else:
            text = "TIME UP"
            color = self.COLOR_TARGET_HIT
            
        game_over_surf = self.font_game_over.render(text, True, color)
        pos_x = self.WIDTH / 2 - game_over_surf.get_width() / 2
        pos_y = self.HEIGHT / 2 - game_over_surf.get_height() / 2
        
        overlay.blit(game_over_surf, (pos_x, pos_y))
        self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "targets_hit": self.num_targets_hit,
            "game_state": self.game_state,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Gravity Ball")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human player ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()