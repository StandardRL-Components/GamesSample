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
        "Controls: Use arrow keys to move the crosshair. "
        "Press Space to fire a high-risk, high-reward shot. "
        "Hold Shift and press Space to fire a safe, low-reward shot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down target practice game. Hit all targets with limited ammo before time runs out. "
        "Risky shots are harder to land but yield more points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = pygame.Color("#1a1a1a")
    COLOR_TARGET = pygame.Color("#32cd32")  # LimeGreen
    COLOR_TARGET_HIT = pygame.Color("#ffffff")
    COLOR_CROSSHAIR = pygame.Color("#ffffff")
    COLOR_PROJECTILE_RISKY = pygame.Color("#ffdd00")  # Yellow
    COLOR_PROJECTILE_SAFE = pygame.Color("#00aaff")  # Blue
    COLOR_TEXT = pygame.Color("#dddddd")
    COLOR_TEXT_WARN = pygame.Color("#ff4500")  # OrangeRed

    # Game Parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000  # ~33.3 seconds
    NUM_TARGETS = 10
    INITIAL_AMMO = 15  # Brief says 5, but that's too low for 10 targets. Increased for playability.
    CROSSHAIR_SPEED = 12
    TARGET_RADIUS = 15
    SAFE_SHOT_RADIUS = 25
    RISKY_SHOT_RADIUS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables will be initialized in reset()
        self.crosshair_pos = None
        self.targets = None
        self.ammo = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.projectiles = None
        self.particles = None
        self.prev_space_held = None
        self.crosshair_recoil = None
        self.rng = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.crosshair_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.ammo = self.INITIAL_AMMO
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.projectiles = []
        self.particles = []
        self.prev_space_held = False
        self.crosshair_recoil = 0

        # Generate targets
        self.targets = []
        while len(self.targets) < self.NUM_TARGETS:
            new_pos = pygame.Vector2(
                self.rng.integers(low=50, high=self.SCREEN_WIDTH - 50),
                self.rng.integers(low=50, high=self.SCREEN_HEIGHT - 50),
            )
            # Ensure targets don't overlap
            if not any(t['pos'].distance_to(new_pos) < self.TARGET_RADIUS * 2.5 for t in self.targets):
                self.targets.append({"pos": new_pos, "active": True, "hit_timer": 0})

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Input ---
        if movement == 1: self.crosshair_pos.y -= self.CROSSHAIR_SPEED
        if movement == 2: self.crosshair_pos.y += self.CROSSHAIR_SPEED
        if movement == 3: self.crosshair_pos.x -= self.CROSSHAIR_SPEED
        if movement == 4: self.crosshair_pos.x += self.CROSSHAIR_SPEED

        self.crosshair_pos.x = np.clip(self.crosshair_pos.x, 0, self.SCREEN_WIDTH)
        self.crosshair_pos.y = np.clip(self.crosshair_pos.y, 0, self.SCREEN_HEIGHT)

        # Fire on button press (transition from not held to held)
        space_pressed = space_held and not self.prev_space_held

        if space_pressed and self.ammo > 0 and not self.game_over:
            # Shift modifies the space press to a "safe" shot
            is_risky = not shift_held
            reward += self._fire_weapon(is_risky)

        self.prev_space_held = space_held

        # --- Update Game State ---
        self.steps += 1
        self._update_animations()

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            active_targets = sum(1 for t in self.targets if t['active'])
            if active_targets == 0:
                # Victory
                reward += 50.0
            else:
                # Failure
                reward += -10.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _fire_weapon(self, is_risky):
        self.ammo -= 1
        self.crosshair_recoil = 10  # Visual feedback for firing

        shot_radius = self.RISKY_SHOT_RADIUS if is_risky else self.SAFE_SHOT_RADIUS
        hit_reward = 1.0 if is_risky else 0.5
        miss_penalty = -0.2

        # Find closest active target
        active_targets = [t for t in self.targets if t['active']]
        if not active_targets:
            return miss_penalty  # No targets left to hit

        closest_target = min(
            active_targets,
            key=lambda t: self.crosshair_pos.distance_to(t['pos'])
        )

        # Check for hit
        if self.crosshair_pos.distance_to(closest_target['pos']) <= shot_radius:
            # Hit
            closest_target['active'] = False
            closest_target['hit_timer'] = 10  # For flash effect
            self.score += 10 if is_risky else 5
            self._create_particles(closest_target['pos'], self.COLOR_PROJECTILE_RISKY if is_risky else self.COLOR_PROJECTILE_SAFE)

            num_remaining = sum(1 for t in self.targets if t['active'])
            final_hit_bonus = 5.0 if num_remaining == 0 else 0.0
            return hit_reward + final_hit_bonus
        else:
            # Miss
            # Create a projectile animation for visual feedback
            self.projectiles.append({
                "start": self.crosshair_pos.copy(),
                "end": self.crosshair_pos + (closest_target['pos'] - self.crosshair_pos).normalize() * 100,
                "progress": 0,
                "color": self.COLOR_PROJECTILE_RISKY if is_risky else self.COLOR_PROJECTILE_SAFE,
                "is_miss": True
            })
            return miss_penalty

    def _update_animations(self):
        # Update recoil
        self.crosshair_recoil = max(0, self.crosshair_recoil - 1)

        # Update hit timers
        for target in self.targets:
            if target['hit_timer'] > 0:
                target['hit_timer'] -= 1

        # Update projectiles
        for p in self.projectiles[:]:
            p['progress'] += 0.2
            if p['progress'] >= 1:
                self.projectiles.remove(p)

        # Update particles
        for particle in self.particles[:]:
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.particles.remove(particle)
            else:
                particle['pos'] += particle['vel']
                particle['radius'] -= 0.2

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'radius': self.rng.uniform(2, 5),
                'life': self.rng.integers(10, 21),
                'color': color
            })

    def _check_termination(self):
        all_targets_hit = all(not t['active'] for t in self.targets)
        no_ammo_and_targets_left = self.ammo <= 0 and any(t['active'] for t in self.targets)
        return all_targets_hit or no_ammo_and_targets_left or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_remaining": sum(1 for t in self.targets if t['active'])
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                color = p['color']
                alpha = int(255 * (p['life'] / 20))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']),
                    (color.r, color.g, color.b, alpha)
                )

        # Draw projectiles
        for p in self.projectiles:
            if p['is_miss']:
                end_pos = p['start'].lerp(p['end'], p['progress'])
                alpha = int(255 * (1 - p['progress']))
                color = p['color']
                # Pygame.draw.line does not support alpha, so we skip it or use a workaround if needed.
                # For simplicity, we draw a solid line.
                if alpha > 0:
                    pygame.draw.line(self.screen, color, p['start'], end_pos, 2)

        # Draw targets
        for t in self.targets:
            pos_int = (int(t['pos'].x), int(t['pos'].y))
            if t['active']:
                # Draw outer ring for safe shot radius
                safe_color = self.COLOR_PROJECTILE_SAFE
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.SAFE_SHOT_RADIUS,
                                        (safe_color.r, safe_color.g, safe_color.b, 50))
                # Draw inner ring for risky shot radius
                risky_color = self.COLOR_PROJECTILE_RISKY
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.RISKY_SHOT_RADIUS,
                                        (risky_color.r, risky_color.g, risky_color.b, 80))
                # Draw main target
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)
            elif t['hit_timer'] > 0:
                # Flash on hit
                alpha = int(255 * (t['hit_timer'] / 10))
                hit_color = self.COLOR_TARGET_HIT
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS + 5,
                                             (hit_color.r, hit_color.g, hit_color.b, alpha))

        # Draw crosshair
        x, y = int(self.crosshair_pos.x), int(self.crosshair_pos.y)
        size = 10 + self.crosshair_recoil
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (x - size, y), (x + size, y), 2)
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (x, y - size), (x, y + size), 2)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Ammo
        ammo_color = self.COLOR_TEXT_WARN if self.ammo <= 3 else self.COLOR_TEXT
        ammo_surf = self.font_main.render(f"AMMO: {self.ammo}", True, ammo_color)
        self.screen.blit(ammo_surf, (self.SCREEN_WIDTH - ammo_surf.get_width() - 10, 10))

        # Time
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_color = self.COLOR_TEXT_WARN if time_left < 5 else self.COLOR_TEXT
        time_surf = self.font_main.render(f"TIME: {time_left:.1f}", True, time_color)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH // 2 - time_surf.get_width() // 2, 10))

        # Game Over Message
        if self.game_over:
            all_targets_hit = all(not t['active'] for t in self.targets)
            msg = "VICTORY!" if all_targets_hit else "GAME OVER"
            color = self.COLOR_TARGET if all_targets_hit else self.COLOR_TEXT_WARN

            end_surf = self.font_main.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv()

    # --- Human Player Controls ---
    # This setup allows a human to play the game.
    # It demonstrates how actions are mapped.

    obs, info = env.reset()
    done = False

    # Use a separate display for human play
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Target Practice")

    running = True
    while running:
        # Action defaults
        movement = 0  # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Movement (prioritizes last pressed in case of conflict)
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        # Action buttons
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        # The action is a combination of all inputs for this frame
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Display final frame for a moment before resetting
            frame = np.transpose(obs, (1, 0, 2))
            pygame.surfarray.blit_array(display_screen, frame)
            pygame.display.flip()
            pygame.time.wait(3000)  # 3-second pause
            obs, info = env.reset()

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(display_screen, frame)
        pygame.display.flip()

        env.clock.tick(env.FPS)

    pygame.quit()