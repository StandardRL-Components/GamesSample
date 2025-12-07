import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:27:59.335541
# Source Brief: brief_00427.md
# Brief Index: 427
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Absorb and match pairs of colored particles to grow larger. "
        "Use portals to teleport and unlock a powerful engulf ability to clear the screen."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to move. Press space to teleport between portals and shift to engulf nearby particles."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (10, 5, 30)
    COLOR_MEMBRANE = (30, 20, 70)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_PORTAL = (120, 0, 255)
    COLOR_PORTAL_GLOW = (60, 0, 128)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_ICON_OFF = (50, 50, 60)
    COLOR_UI_ICON_ON = (0, 255, 128)

    PARTICLE_COLORS = [
        (255, 0, 128),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 0),  # Green
        (255, 128, 0),  # Orange
        (200, 0, 255),  # Purple
    ]

    # Game Parameters
    PLAYER_SPEED = 4.0
    INITIAL_PLAYER_RADIUS = 12
    PARTICLE_RADIUS = 6
    PARTICLE_COUNT_PER_TYPE = 5
    MAX_STEPS = 1000
    MATCH_DISTANCE_FACTOR = 1.5

    PORTAL_COOLDOWN = 60  # frames
    ENGULF_COOLDOWN = 90  # frames

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_radius = None
        self.target_player_radius = None
        self.particles = None
        self.portals = None
        self.explosions = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.engulf_unlocked = None
        self.engulf_radius = None
        self.engulf_level = None
        self.portal_cooldown_timer = None
        self.engulf_cooldown_timer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_radius = self.INITIAL_PLAYER_RADIUS
        self.target_player_radius = self.INITIAL_PLAYER_RADIUS

        self._generate_particles()

        self.portals = [
            np.array([80, 80], dtype=np.float32),
            np.array([self.SCREEN_WIDTH - 80, self.SCREEN_HEIGHT - 80], dtype=np.float32)
        ]

        self.explosions = []

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.engulf_unlocked = False
        self.engulf_radius = 0
        self.engulf_level = 0
        self.portal_cooldown_timer = 0
        self.engulf_cooldown_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # --- 1. Update Cooldowns ---
        self.portal_cooldown_timer = max(0, self.portal_cooldown_timer - 1)
        self.engulf_cooldown_timer = max(0, self.engulf_cooldown_timer - 1)

        # --- 2. Handle Input and Movement ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        dist_before = self._find_nearest_particle_dist()

        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1  # Up
        elif movement == 2: move_vec[1] = 1  # Down
        elif movement == 3: move_vec[0] = -1  # Left
        elif movement == 4: move_vec[0] = 1  # Right

        if np.linalg.norm(move_vec) > 0:
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.SCREEN_WIDTH - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.SCREEN_HEIGHT - self.player_radius)

        dist_after = self._find_nearest_particle_dist()

        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 1.0  # Moved closer to nearest particle
            elif dist_after > dist_before:
                reward -= 0.1  # Moved away from nearest particle

        # --- 3. Handle Actions ---
        # Portal Action
        if space_held and self.portal_cooldown_timer == 0:
            # Find which portal is closer and teleport to the other one
            dist_to_portal0 = np.linalg.norm(self.player_pos - self.portals[0])
            dist_to_portal1 = np.linalg.norm(self.player_pos - self.portals[1])
            if min(dist_to_portal0, dist_to_portal1) < 40:  # Must be close to a portal
                if dist_to_portal0 < dist_to_portal1:
                    self.player_pos = self.portals[1].copy()
                else:
                    self.player_pos = self.portals[0].copy()
                self.portal_cooldown_timer = self.PORTAL_COOLDOWN
                # SFX: Portal whoosh
                self._add_explosion(self.player_pos, self.COLOR_PORTAL, 50)

        # Engulf Action
        if shift_held and self.engulf_unlocked and self.engulf_cooldown_timer == 0:
            consumed_in_engulf = []
            for p in self.particles:
                if np.linalg.norm(self.player_pos - p['pos']) < self.engulf_radius:
                    consumed_in_engulf.append(p)

            if consumed_in_engulf:
                reward += 20.0  # Effective use of ability
                for p in consumed_in_engulf:
                    self.score += 1
                    self._add_explosion(p['pos'], p['color'], 30)
                    # SFX: Power-up consume
                self.particles = [p for p in self.particles if p not in consumed_in_engulf]
            self.engulf_cooldown_timer = self.ENGULF_COOLDOWN
            # SFX: Engulf activation
            self._add_explosion(self.player_pos, self.COLOR_UI_ICON_ON, self.engulf_radius, life=20)

        # --- 4. Handle Automatic Matching ---
        adjacent_particles = []
        for p in self.particles:
            dist = np.linalg.norm(self.player_pos - p['pos'])
            if dist < self.player_radius + self.PARTICLE_RADIUS * self.MATCH_DISTANCE_FACTOR:
                adjacent_particles.append(p)

        if len(adjacent_particles) >= 2:
            by_color = {}
            for p in adjacent_particles:
                color_key = tuple(p['color'])
                if color_key not in by_color:
                    by_color[color_key] = []
                by_color[color_key].append(p)

            consumed_this_step = []
            for color, group in by_color.items():
                if len(group) >= 2:
                    p1, p2 = group[0], group[1]
                    consumed_this_step.extend([p1, p2])
                    self.score += 2  # 1 per particle
                    reward += 10.0  # Match bonus
                    # SFX: Match success
                    mid_point = (p1['pos'] + p2['pos']) / 2
                    self._add_explosion(mid_point, p1['color'], 40)

            if consumed_this_step:
                consumed_ids = {p['id'] for p in consumed_this_step}
                self.particles = [p for p in self.particles if p['id'] not in consumed_ids]

        # --- 5. Update World State ---
        # Particle movement
        for p in self.particles:
            p['pos'] += p['vel']
            if p['pos'][0] < self.PARTICLE_RADIUS or p['pos'][0] > self.SCREEN_WIDTH - self.PARTICLE_RADIUS:
                p['vel'][0] *= -1
            if p['pos'][1] < self.PARTICLE_RADIUS or p['pos'][1] > self.SCREEN_HEIGHT - self.PARTICLE_RADIUS:
                p['vel'][1] *= -1

        # Update effects
        self.explosions = [e for e in self.explosions if e['life'] > 0]
        for e in self.explosions:
            e['life'] -= 1
            e['radius'] += e['expand_rate']

        # Check for ability unlocks
        if self.score >= 25 and self.engulf_level == 0:
            self.engulf_unlocked = True
            self.engulf_radius = 60
            self.engulf_level = 1
            # SFX: Upgrade unlock
        if self.score >= 50 and self.engulf_level == 1:
            self.engulf_radius = 80
            self.engulf_level = 2
            # SFX: Upgrade unlock
        if self.score >= 75 and self.engulf_level == 2:
            self.engulf_radius = 100
            self.engulf_level = 3
            # SFX: Upgrade unlock

        # Update player size based on score
        self.target_player_radius = self.INITIAL_PLAYER_RADIUS + math.sqrt(self.score) * 2
        self.player_radius += (self.target_player_radius - self.player_radius) * 0.1  # Smooth interpolation

        # --- 6. Finalize Step ---
        terminated = (len(self.particles) == 0) or (self.steps >= self.MAX_STEPS)
        if terminated and len(self.particles) == 0:
            reward += 100.0  # Victory bonus

        self.game_over = terminated
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_particles(self):
        self.particles = []
        particle_id = 0
        for color in self.PARTICLE_COLORS:
            for _ in range(self.PARTICLE_COUNT_PER_TYPE):
                for _ in range(2):  # Create a pair
                    placed = False
                    while not placed:
                        pos = np.array([
                            self.np_random.uniform(self.PARTICLE_RADIUS, self.SCREEN_WIDTH - self.PARTICLE_RADIUS),
                            self.np_random.uniform(self.PARTICLE_RADIUS, self.SCREEN_HEIGHT - self.PARTICLE_RADIUS)
                        ])
                        # Ensure not spawning on other particles or player
                        if np.linalg.norm(pos - self.player_pos) < 50: continue

                        too_close = False
                        for p in self.particles:
                            if np.linalg.norm(pos - p['pos']) < self.PARTICLE_RADIUS * 3:
                                too_close = True
                                break
                        if not too_close:
                            placed = True

                    vel = self.np_random.uniform(-0.5, 0.5, size=2)
                    self.particles.append({'id': particle_id, 'pos': pos, 'vel': vel, 'color': color})
                    particle_id += 1

    def _find_nearest_particle_dist(self):
        if not self.particles:
            return None
        distances = [np.linalg.norm(self.player_pos - p['pos']) for p in self.particles]
        return min(distances)

    def _add_explosion(self, pos, color, max_radius, life=30):
        self.explosions.append({
            'pos': pos.copy(),
            'color': color,
            'radius': 0,
            'max_radius': max_radius,
            'life': life,
            'max_life': life,
            'expand_rate': max_radius / life
        })

    def _get_observation(self):
        # Fill background and draw membrane
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_MEMBRANE, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_MEMBRANE, (0, i), (self.SCREEN_WIDTH, i), 1)
        pygame.draw.rect(self.screen, self.COLOR_MEMBRANE, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 5)

        # Draw portals
        for i, p_pos in enumerate(self.portals):
            pulse_rad = 30 + 5 * math.sin(pygame.time.get_ticks() * 0.002 + i * math.pi)
            self._draw_glow_circle(self.screen, self.COLOR_PORTAL, self.COLOR_PORTAL_GLOW, p_pos, pulse_rad, 10)

        # Draw particles
        for p in self.particles:
            pos_int = p['pos'].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PARTICLE_RADIUS, p['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PARTICLE_RADIUS, p['color'])

        # Draw player
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, self.player_pos, self.player_radius,
                               20)

        # Draw effects
        for e in self.explosions:
            pos_int = e['pos'].astype(int)
            progress = e['life'] / e['max_life']
            current_alpha = int(200 * progress)
            radius = int(e['radius'])

            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*e['color'], current_alpha), (radius, radius), radius)
                self.screen.blit(temp_surf, (pos_int[0] - radius, pos_int[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score and step text
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        particles_left = len(self.particles)
        particles_text = self.font_small.render(f"TARGETS: {particles_left}", True, self.COLOR_TEXT)
        self.screen.blit(particles_text, (15, 35))

        step_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (self.SCREEN_WIDTH - step_text.get_width() - 15, 10))

        # Skill tree UI
        ui_y = 20
        # Portal ability
        portal_color = self.COLOR_UI_ICON_ON if self.portal_cooldown_timer == 0 else self.COLOR_UI_ICON_OFF
        pygame.draw.rect(self.screen, portal_color, (self.SCREEN_WIDTH - 150, ui_y, 20, 20), border_radius=4)
        portal_label = self.font_small.render("PORTAL", True, self.COLOR_TEXT)
        self.screen.blit(portal_label, (self.SCREEN_WIDTH - 120, ui_y))

        # Engulf ability
        engulf_color = self.COLOR_UI_ICON_OFF
        if self.engulf_unlocked:
            engulf_color = self.COLOR_UI_ICON_ON if self.engulf_cooldown_timer == 0 else self.COLOR_UI_ICON_OFF

        pygame.draw.rect(self.screen, engulf_color, (self.SCREEN_WIDTH - 150, ui_y + 25, 20, 20), border_radius=4)
        engulf_label = self.font_small.render("ENGULF", True, self.COLOR_TEXT)
        self.screen.blit(engulf_label, (self.SCREEN_WIDTH - 120, ui_y + 25))

        # Engulf level indicators
        for i in range(3):
            level_color = self.COLOR_UI_ICON_ON if self.engulf_level > i else self.COLOR_UI_ICON_OFF
            pygame.draw.circle(self.screen, level_color, (self.SCREEN_WIDTH - 50 + i * 15, ui_y + 35), 4)

    def _draw_glow_circle(self, surface, color, glow_color, center, radius, glow_strength):
        center_int = center.astype(int)
        radius = int(radius)
        if radius <= 0: return

        for i in range(glow_strength, 0, -1):
            alpha = int(100 * (1 - i / glow_strength))
            glow_rad = radius + i

            temp_surf = pygame.Surface((glow_rad * 2, glow_rad * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*glow_color, alpha), (glow_rad, glow_rad), glow_rad)
            surface.blit(temp_surf, (center_int[0] - glow_rad, center_int[1] - glow_rad),
                         special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "particles_left": len(self.particles),
            "engulf_level": self.engulf_level,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Phagocyte")
    clock = pygame.time.Clock()

    total_reward = 0

    terminated = False
    truncated = False
    while not (terminated or truncated):
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True # End the loop if window is closed

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False


        clock.tick(30)  # Run at 30 FPS

    env.close()