import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade-style Gymnasium environment where the player controls swinging pendulums
    to launch colored orbs into matching target zones.

    The game prioritizes visual appeal and "game feel" with smooth animations,
    particle effects, and responsive controls.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        - Up/Down: Control left pendulum's swing speed.
        - Left/Right: Control right pendulum's swing speed.
    - action[1]: Space button (0=released, 1=held) - Launches an orb.
    - action[2]: Shift button (0=released, 1=held) - No effect.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +1 for hitting the correct target.
    - +10 for completing a level.
    - +100 for winning the game (completing all 3 levels).
    - +/-0.01 shaping reward for orb moving towards/away from its target.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Control swinging pendulums to launch colored orbs into their matching target zones."
    )
    user_guide = (
        "Use ↑/↓ to control the left pendulum and ←/→ for the right. Press space to launch an orb."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2500

        # Colors
        self.COLOR_BG = (15, 19, 32)
        self.COLOR_PENDULUM = (100, 110, 140)
        self.COLOR_PIVOT = (180, 190, 220)
        self.COLOR_TEXT = (220, 220, 240)
        self.TARGET_COLORS = [
            (255, 70, 70),   # Red
            (70, 255, 70),   # Green
            (70, 130, 255),  # Blue
            (255, 200, 70),  # Yellow
        ]
        self.NUM_PENDULUMS = 2

        # Physics & Gameplay
        self.GRAVITY = 0.4
        self.ORB_RADIUS = 10
        self.LAUNCH_POWER = 1.2
        self.LEVEL_GOALS = {1: 15, 2: 25, 3: 35}
        self.SPEED_UP_INTERVAL = 5
        self.SPEED_UP_AMOUNT = 0.15
        self.MIN_SWING_SPEED = 0.01
        self.MAX_SWING_SPEED = 0.1
        self.SWING_ADJUST_RATE = 0.002

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("monospace", 18, bold=True)
        self.level_font = pygame.font.SysFont("monospace", 24, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.total_hits = 0
        self.hits_this_level = 0
        self.speed_multiplier = 1.0
        self.pendulums = []
        self.targets = []
        self.active_orb = None
        self.next_orb_color_idx = 0
        self.particles = []
        self.last_adjusted_pendulum_idx = 0
        self.space_was_held = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.level = 1
        self.total_hits = 0
        self.hits_this_level = 0
        self.speed_multiplier = 1.0

        self._create_pendulums()
        self._create_targets()

        self.active_orb = None
        self.next_orb_color_idx = self.np_random.integers(0, len(self.TARGET_COLORS))

        self.particles = []
        self.last_adjusted_pendulum_idx = 0
        self.space_was_held = True  # Prevent launch on first frame

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        self._handle_input(action)
        self._update_game_state()

        reward = self._handle_events_and_calc_reward()
        self.score += reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Adjust pendulum speeds
        if movement == 1:  # Up
            self.pendulums[0]['swing_speed'] += self.SWING_ADJUST_RATE
            self.last_adjusted_pendulum_idx = 0
        elif movement == 2:  # Down
            self.pendulums[0]['swing_speed'] -= self.SWING_ADJUST_RATE
            self.last_adjusted_pendulum_idx = 0
        elif movement == 4:  # Right
            self.pendulums[1]['swing_speed'] += self.SWING_ADJUST_RATE
            self.last_adjusted_pendulum_idx = 1
        elif movement == 3:  # Left
            self.pendulums[1]['swing_speed'] -= self.SWING_ADJUST_RATE
            self.last_adjusted_pendulum_idx = 1

        for p in self.pendulums:
            p['swing_speed'] = np.clip(p['swing_speed'], self.MIN_SWING_SPEED, self.MAX_SWING_SPEED)

        # Launch orb on space press (rising edge)
        if space_held and not self.space_was_held:
            self._launch_orb()
        self.space_was_held = space_held

    def _update_game_state(self):
        self._update_pendulums()
        self._update_orb()
        self._update_particles()

    def _handle_events_and_calc_reward(self):
        reward = 0

        if self.active_orb is None:
            return 0

        # Orb-target collision
        orb_rect = pygame.Rect(self.active_orb['pos'][0] - self.ORB_RADIUS, self.active_orb['pos'][1] - self.ORB_RADIUS, self.ORB_RADIUS * 2, self.ORB_RADIUS * 2)

        hit = False
        for i, target in enumerate(self.targets):
            if orb_rect.colliderect(target['rect']):
                if self.active_orb['color_idx'] == target['color_idx']:
                    reward += 1.0  # Hit reward
                    self.total_hits += 1
                    self.hits_this_level += 1
                    self._create_particles(self.active_orb['pos'], self.TARGET_COLORS[self.active_orb['color_idx']], 50)

                    # Check for speed up
                    if self.total_hits > 0 and self.total_hits % self.SPEED_UP_INTERVAL == 0:
                        self.speed_multiplier += self.SPEED_UP_AMOUNT

                    # Check for level up
                    if self.hits_this_level >= self.LEVEL_GOALS[self.level]:
                        self.level += 1
                        if self.level <= 3:
                            reward += 10.0  # Level complete reward
                            self.hits_this_level = 0
                        else:
                            # Game won
                            reward += 100.0
                            self.game_over = True
                    hit = True
                else:
                    reward -= 0.5 # Penalty for hitting wrong target
                    self._create_particles(self.active_orb['pos'], (150, 150, 150), 20)
                    hit = True
                break

        if hit:
            self.active_orb = None
            self.next_orb_color_idx = self.np_random.integers(0, len(self.TARGET_COLORS))
            return reward

        # Shaping reward
        target_center = self.targets[self.active_orb['color_idx']]['rect'].center
        current_dist = math.hypot(self.active_orb['pos'][0] - target_center[0], self.active_orb['pos'][1] - target_center[1])
        if self.active_orb['last_dist'] is not None:
            dist_delta = self.active_orb['last_dist'] - current_dist
            reward += np.clip(dist_delta * 0.01, -0.01, 0.01) # Small shaping reward
        self.active_orb['last_dist'] = current_dist

        return reward

    def _check_termination(self):
        if self.game_over: # Game won
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "total_hits": self.total_hits,
            "hits_this_level": self.hits_this_level
        }

    # --- Helper methods for creation ---
    def _create_pendulums(self):
        self.pendulums = []
        positions = [self.WIDTH * 0.25, self.WIDTH * 0.75]
        for i in range(self.NUM_PENDULUMS):
            self.pendulums.append({
                'pivot': (positions[i], 50),
                'length': 80,
                'angle': 0.0,
                'angular_velocity': 0.0,
                'swing_speed': self.np_random.uniform(0.02, 0.04),
            })

    def _create_targets(self):
        self.targets = []
        target_width, target_height = 80, 20
        y_pos = self.HEIGHT - 40
        positions = [
            (self.WIDTH * 0.1, y_pos), (self.WIDTH * 0.35, y_pos),
            (self.WIDTH * 0.6, y_pos), (self.WIDTH * 0.85, y_pos)
        ]
        color_indices = list(range(len(self.TARGET_COLORS)))
        self.np_random.shuffle(color_indices)
        for i in range(len(self.TARGET_COLORS)):
            self.targets.append({
                'rect': pygame.Rect(positions[i][0] - target_width / 2, positions[i][1], target_width, target_height),
                'color_idx': color_indices[i]
            })

    def _launch_orb(self):
        if self.active_orb is not None:
            return

        p = self.pendulums[self.last_adjusted_pendulum_idx]

        # Orb starts at the tip of the pendulum
        start_pos_x = p['pivot'][0] + p['length'] * math.sin(p['angle'])
        start_pos_y = p['pivot'][1] + p['length'] * math.cos(p['angle'])

        # Orb velocity is derived from pendulum's tangential velocity
        tangential_speed = p['angular_velocity'] * p['length'] * self.LAUNCH_POWER
        vel_x = tangential_speed * math.cos(p['angle'])
        vel_y = tangential_speed * -math.sin(p['angle'])

        self.active_orb = {
            'pos': [start_pos_x, start_pos_y],
            'vel': [vel_x, vel_y],
            'color_idx': self.next_orb_color_idx,
            'last_dist': None
        }

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifetime': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    # --- Helper methods for updates ---
    def _update_pendulums(self):
        for p in self.pendulums:
            # Simple harmonic motion approximation
            acceleration = -p['swing_speed'] * math.sin(p['angle']) * self.speed_multiplier
            p['angular_velocity'] += acceleration
            p['angular_velocity'] *= 0.99  # Damping
            p['angle'] += p['angular_velocity']
            p['angle'] = np.clip(p['angle'], -math.pi/1.5, math.pi/1.5)

    def _update_orb(self):
        if self.active_orb is None:
            return

        self.active_orb['vel'][1] += self.GRAVITY
        self.active_orb['pos'][0] += self.active_orb['vel'][0]
        self.active_orb['pos'][1] += self.active_orb['vel'][1]

        # Remove orb if it goes off-screen
        x, y = self.active_orb['pos']
        if not (-self.ORB_RADIUS < x < self.WIDTH + self.ORB_RADIUS and y < self.HEIGHT + self.ORB_RADIUS):
            self.active_orb = None
            self.next_orb_color_idx = self.np_random.integers(0, len(self.TARGET_COLORS))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98
            p['vel'][1] *= 0.98
            p['lifetime'] -= 1

    # --- Helper methods for rendering ---
    def _render_game(self):
        self._render_targets()
        self._render_pendulums()
        self._render_orb_preview()
        self._render_orb()
        self._render_particles()

    def _render_targets(self):
        for target in self.targets:
            color = self.TARGET_COLORS[target['color_idx']]
            pygame.draw.rect(self.screen, color, target['rect'], border_radius=5)
            # Glow effect
            glow_rect = target['rect'].inflate(10, 10)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*color, 50), glow_surface.get_rect(), border_radius=8)
            self.screen.blit(glow_surface, glow_rect.topleft)

    def _render_pendulums(self):
        for i, p in enumerate(self.pendulums):
            px, py = p['pivot']
            end_x = int(px + p['length'] * math.sin(p['angle']))
            end_y = int(py + p['length'] * math.cos(p['angle']))

            # Rod
            pygame.gfxdraw.line(self.screen, int(px), int(py), end_x, end_y, self.COLOR_PENDULUM)

            # Pivot
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 6, self.COLOR_PIVOT)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), 6, self.COLOR_PIVOT)

            # Bob
            pygame.gfxdraw.filled_circle(self.screen, end_x, end_y, 8, self.COLOR_PENDULUM)
            pygame.gfxdraw.aacircle(self.screen, end_x, end_y, 8, self.COLOR_PENDULUM)

            # Highlight last adjusted
            if i == self.last_adjusted_pendulum_idx:
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), 12, (255, 255, 255, 80))

    def _render_orb(self):
        if self.active_orb:
            color = self.TARGET_COLORS[self.active_orb['color_idx']]
            pos = (int(self.active_orb['pos'][0]), int(self.active_orb['pos'][1]))
            self._render_glow_circle(self.screen, color, pos, self.ORB_RADIUS, 3)

    def _render_orb_preview(self):
        if self.active_orb is None:
            p = self.pendulums[self.last_adjusted_pendulum_idx]
            color = self.TARGET_COLORS[self.next_orb_color_idx]

            end_x = int(p['pivot'][0] + p['length'] * math.sin(p['angle']))
            end_y = int(p['pivot'][1] + p['length'] * math.cos(p['angle']))

            # Pulsing alpha for preview
            alpha = int(100 + 50 * math.sin(self.steps * 0.2))
            temp_surface = pygame.Surface((self.ORB_RADIUS*2, self.ORB_RADIUS*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, self.ORB_RADIUS, self.ORB_RADIUS, self.ORB_RADIUS, (*color, alpha))
            pygame.gfxdraw.aacircle(temp_surface, self.ORB_RADIUS, self.ORB_RADIUS, self.ORB_RADIUS, (*color, alpha))
            self.screen.blit(temp_surface, (end_x - self.ORB_RADIUS, end_y - self.ORB_RADIUS))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, p['lifetime'] * 8)
            color_with_alpha = (*p['color'], alpha)
            size = int(p['size'] * (p['lifetime'] / 30.0))
            if size > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                # Use a surface for alpha blending
                temp_surface = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surface, color_with_alpha, (size, size), size)
                self.screen.blit(temp_surface, (pos[0]-size, pos[1]-size))


    def _render_ui(self):
        # Level and Hits
        level_text = self.level_font.render(f"LEVEL {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (20, 15))

        hits_needed = self.LEVEL_GOALS.get(self.level, "-")
        hits_text = self.ui_font.render(f"HITS: {self.hits_this_level} / {hits_needed}", True, self.COLOR_TEXT)
        self.screen.blit(hits_text, (20, 45))

        # Score
        score_text = self.ui_font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            win_status = "YOU WIN!" if self.level > 3 else "TIME UP"
            end_text = self.level_font.render(win_status, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.ui_font.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _render_glow_circle(self, surface, color, center, radius, glow_layers=5):
        for i in range(glow_layers, 0, -1):
            alpha = int(80 * (1 - i / glow_layers))
            current_radius = int(radius + i * 2)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], current_radius, (*color, alpha))
            pygame.gfxdraw.aacircle(surface, center[0], center[1], current_radius, (*color, alpha))

        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0

    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pendulum Orb Guider")
    clock = pygame.time.Clock()

    # Control mapping
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        # --- Human Input ---
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key

        if keys[pygame.K_SPACE]:
            space_held = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()
    pygame.quit()