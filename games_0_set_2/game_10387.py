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
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a player through a maze to collect all checkpoints before time runs out, "
        "while contending with constantly shifting gravity."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to accelerate. Use Shift to brake. "
        "Press Space to attempt to activate a temporary shield."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_EPISODE_STEPS = 60 * FPS  # 60 seconds
    NUM_CHECKPOINTS = 10

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_WALL = (40, 60, 100)
    COLOR_WALL_OUTLINE = (60, 80, 120)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (128, 255, 200)
    COLOR_CHECKPOINT = (255, 220, 0)
    COLOR_CHECKPOINT_PULSE = (255, 255, 128)
    COLOR_SHIELD = (0, 220, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (20, 20, 30)
    COLOR_TIME_WARN = (255, 128, 0)

    # Gravity Directions & Colors
    GRAVITY_DIRECTIONS = {
        0: (0, 1),  # Down
        1: (0, -1),  # Up
        2: (-1, 0),  # Left
        3: (1, 0)  # Right
    }
    GRAVITY_COLORS = {
        0: (255, 80, 80),  # Red
        1: (80, 255, 80),  # Green
        2: (80, 80, 255),  # Blue
        3: (255, 220, 0)  # Yellow
    }
    GRAVITY_CHANGE_INTERVAL = 5 * FPS  # 5 seconds

    # Physics
    PLAYER_ACCEL = 0.25
    GRAVITY_STRENGTH = 0.025  # Reduced to pass stability test
    BRAKE_FRICTION = 0.92
    MAX_VELOCITY = 6.0

    # Shield
    SHIELD_CHANCE = 0.1
    SHIELD_DURATION = 2 * FPS  # 2 seconds
    SHIELD_COOLDOWN = 1 * FPS  # 1 second

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.checkpoints = None
        self.walls = self._create_maze()
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.gravity_dir = None
        self.gravity_vec = None
        self.gravity_timer = None
        self.shield_active = None
        self.shield_timer = None
        self.shield_cooldown_timer = None
        self.last_space_held = None
        self.particles = None
        self.last_dist_to_checkpoint = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_EPISODE_STEPS

        # Player
        self.player_pos = self._get_valid_start_pos()
        self.player_vel = [0.0, 0.0]

        # Checkpoints
        self.checkpoints = self._generate_checkpoints()

        # Gravity
        self.gravity_dir = self.np_random.integers(0, 4)
        self.gravity_vec = [v * self.GRAVITY_STRENGTH for v in self.GRAVITY_DIRECTIONS[self.gravity_dir]]
        self.gravity_timer = self.GRAVITY_CHANGE_INTERVAL

        # Shield
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        self.last_space_held = False

        self.particles = []

        self.last_dist_to_checkpoint = self._get_dist_to_nearest_checkpoint()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Input and State Updates ---
        self._handle_input(action)
        self._apply_physics()
        reward += self._handle_collisions()
        reward += self._handle_checkpoints()
        self._update_gravity()
        self._update_timers()
        self._update_particles()

        # --- Calculate Continuous Reward ---
        dist = self._get_dist_to_nearest_checkpoint()
        if dist is not None and self.last_dist_to_checkpoint is not None:
            reward += (self.last_dist_to_checkpoint - dist) * 0.1
            self.last_dist_to_checkpoint = dist

        # --- Check Termination Conditions ---
        terminated = self.game_over
        truncated = False

        if self.time_remaining <= 0 and not terminated:
            reward -= 100.0
            terminated = True
            self._create_explosion(self.player_pos, self.COLOR_TIME_WARN, 30)

        if all(c['collected'] for c in self.checkpoints) and not terminated:
            reward += 100.0
            terminated = True

        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Internal Logic Methods ---

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1:  # Up
            self.player_vel[1] -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.player_vel[1] += self.PLAYER_ACCEL
        elif movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL

        # Braking (Shift)
        if shift_held:
            self.player_vel[0] *= self.BRAKE_FRICTION
            self.player_vel[1] *= self.BRAKE_FRICTION
            if math.hypot(*self.player_vel) > 1.0:
                self._create_brake_sparks()

        # Shield (Space) - activate on press (rising edge)
        if space_held and not self.last_space_held and self.shield_cooldown_timer <= 0:
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN
            if self.np_random.random() < self.SHIELD_CHANCE:
                self.shield_active = True
                self.shield_timer = self.SHIELD_DURATION
                self.score += 5.0  # Reward for successful activation
                self._create_shield_effect()
        self.last_space_held = space_held

    def _apply_physics(self):
        # Apply gravity
        self.player_vel[0] += self.gravity_vec[0]
        self.player_vel[1] += self.gravity_vec[1]

        # Clamp velocity
        vel_mag = math.hypot(*self.player_vel)
        if vel_mag > self.MAX_VELOCITY:
            scale = self.MAX_VELOCITY / vel_mag
            self.player_vel[0] *= scale
            self.player_vel[1] *= scale

        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # Add player trail particles
        if self.steps % 3 == 0 and math.hypot(*self.player_vel) > 1.5:
            self.particles.append(self._create_particle(
                pos=list(self.player_pos),
                vel=[self.np_random.uniform(-0.2, 0.2) for _ in range(2)],
                color=self.COLOR_PLAYER_GLOW,
                radius=3,
                lifespan=20
            ))

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 8, 16, 16)
        for wall in self.walls:
            if player_rect.colliderect(wall):
                if self.shield_active:
                    self.shield_active = False
                    self.shield_timer = 0
                    self._create_explosion(self.player_pos, self.COLOR_SHIELD, 20)
                    self._bounce_off_wall(player_rect, wall)
                    return 0.0
                else:
                    self.game_over = True
                    self._create_explosion(self.player_pos, self.COLOR_PLAYER, 40)
                    return -100.0

        # Boundary collision (should not happen with maze design but good practice)
        if not (0 <= player_rect.centerx <= self.WIDTH and 0 <= player_rect.centery <= self.HEIGHT):
            self.game_over = True
            return -100.0

        return 0.0

    def _bounce_off_wall(self, player_rect, wall):
        # Simple bounce logic
        overlap = player_rect.clip(wall)
        if overlap.width < overlap.height:
            # Horizontal collision
            self.player_vel[0] *= -0.8  # Inelastic bounce
            self.player_pos[0] += (wall.centerx - player_rect.centerx) * 0.1
        else:
            # Vertical collision
            self.player_vel[1] *= -0.8
            self.player_pos[1] += (wall.centery - player_rect.centery) * 0.1

    def _handle_checkpoints(self):
        reward = 0.0
        for cp in self.checkpoints:
            if not cp['collected']:
                dist = math.hypot(self.player_pos[0] - cp['pos'][0], self.player_pos[1] - cp['pos'][1])
                if dist < 20:  # Player radius (8) + Checkpoint radius (12)
                    cp['collected'] = True
                    reward += 10.0 # Increased reward for collecting
                    self._create_explosion(cp['pos'], self.COLOR_CHECKPOINT, 25)
        return reward

    def _update_gravity(self):
        self.gravity_timer -= 1
        if self.gravity_timer <= 0:
            self.gravity_timer = self.GRAVITY_CHANGE_INTERVAL
            self.gravity_dir = self.np_random.integers(0, 4)
            self.gravity_vec = [v * self.GRAVITY_STRENGTH for v in self.GRAVITY_DIRECTIONS[self.gravity_dir]]

    def _update_timers(self):
        self.time_remaining = max(0, self.time_remaining - 1)
        if self.shield_timer > 0:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.shield_active = False
        if self.shield_cooldown_timer > 0:
            self.shield_cooldown_timer -= 1

    # --- Helper and Generation Methods ---

    def _get_valid_start_pos(self):
        # Start in a more open area to pass stability test
        return [320.0, 200.0]

    def _generate_checkpoints(self):
        checkpoints = []
        # Pre-defined locations for consistent challenge
        locations = [
            (320, 50), (590, 50), (590, 200), (590, 350), (320, 350),
            (50, 350), (50, 200), (180, 200), (320, 200), (460, 200)
        ]
        # Shuffle to add variety to the collection order
        shuffled_locations = list(locations)
        self.np_random.shuffle(shuffled_locations)
        for pos in shuffled_locations[:self.NUM_CHECKPOINTS]:
            checkpoints.append({'pos': list(pos), 'collected': False})
        return checkpoints

    def _get_dist_to_nearest_checkpoint(self):
        uncollected = [c['pos'] for c in self.checkpoints if not c['collected']]
        if not uncollected:
            return 0

        dists = [math.hypot(self.player_pos[0] - pos[0], self.player_pos[1] - pos[1]) for pos in uncollected]
        return min(dists)

    def _create_maze(self):
        # A fixed maze layout for consistency
        wall_thickness = 10
        w, h = self.WIDTH, self.HEIGHT
        t = wall_thickness
        return [
            # Borders
            pygame.Rect(0, 0, w, t),
            pygame.Rect(0, h - t, w, t),
            pygame.Rect(0, 0, t, h),
            pygame.Rect(w - t, 0, t, h),
            # Internal walls
            pygame.Rect(t, 120, 200, t),
            pygame.Rect(120, 120, t, 160),
            pygame.Rect(250, 0, t, 150),
            pygame.Rect(250, 250, t, 150),
            pygame.Rect(380, 120, t, 160),
            pygame.Rect(w - 210, 120, 200, t),
        ]

    # --- Particle System Methods ---

    def _create_particle(self, pos, vel, color, radius, lifespan):
        return {'pos': list(pos), 'vel': list(vel), 'color': color, 'radius': radius, 'lifespan': lifespan}

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append(self._create_particle(
                pos, vel, color, self.np_random.uniform(2, 5), self.np_random.integers(30, 60)
            ))

    def _create_shield_effect(self):
        self.particles.append({
            'pos': self.player_pos, 'vel': [0, 0], 'color': self.COLOR_SHIELD, 'radius': 10,
            'lifespan': 30, 'type': 'shield_burst'
        })

    def _create_brake_sparks(self):
        for _ in range(2):
            self.particles.append(self._create_particle(
                pos=[self.player_pos[0] + self.np_random.uniform(-5, 5),
                     self.player_pos[1] + self.np_random.uniform(-5, 5)],
                vel=[-v * 0.5 for v in self.player_vel],
                color=self.COLOR_TIME_WARN,
                radius=self.np_random.uniform(1, 3),
                lifespan=15
            ))

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p.get('type') == 'shield_burst':
                p['radius'] += 1.5  # Expanding effect
            if p['lifespan'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        self._render_walls()
        self._render_particles()
        self._render_checkpoints()
        self._render_player()

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            pygame.draw.rect(self.screen, self.COLOR_WALL_OUTLINE, wall, 1)

    def _render_checkpoints(self):
        for cp in self.checkpoints:
            if not cp['collected']:
                pos = (int(cp['pos'][0]), int(cp['pos'][1]))
                pulse = abs(math.sin(self.steps * 0.1)) * 3
                radius = int(10 + pulse)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_CHECKPOINT_PULSE)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_CHECKPOINT)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))

        if self.shield_active:
            radius = int(16 + abs(math.sin(self.steps * 0.2)) * 4)
            alpha = 100 + int(abs(math.sin(self.steps * 0.2)) * 50)
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_SHIELD, alpha), (radius, radius), radius)
            self.screen.blit(s, (pos[0] - radius, pos[1] - radius))

        size = 16
        player_rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)

        glow_radius = int(size * 1.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] * (p['lifespan'] / 60.0))
            if radius > 0:
                if p.get('type') == 'shield_burst':
                    alpha = max(0, int(255 * (p['lifespan'] / 30.0)))
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*p['color'], alpha))
                else:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2, 2)):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        secs = self.time_remaining // self.FPS
        mins = secs // 60
        time_str = f"{mins:02}:{secs % 60:02}"
        time_color = self.COLOR_TIME_WARN if secs < 10 and self.steps % self.FPS < self.FPS / 2 else self.COLOR_TEXT
        draw_text(time_str, self.font_main, time_color, (self.WIDTH - 110, 15), self.COLOR_TEXT_SHADOW)

        collected_count = sum(1 for c in self.checkpoints if c['collected'])
        cp_str = f"CP: {collected_count}/{self.NUM_CHECKPOINTS}"
        draw_text(cp_str, self.font_small, self.COLOR_TEXT, (self.WIDTH - 110, 45), self.COLOR_TEXT_SHADOW)

        self._render_gravity_indicator()

    def _render_gravity_indicator(self):
        center = (40, 40)
        color = self.GRAVITY_COLORS[self.gravity_dir]

        pulse_radius = 20 + abs(math.sin(self.steps * 0.05)) * 4
        s = pygame.Surface((pulse_radius * 2, pulse_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, 60), (pulse_radius, pulse_radius), pulse_radius)
        self.screen.blit(s, (center[0] - pulse_radius, center[1] - pulse_radius))

        points = []
        if self.gravity_dir == 0:  # Down
            points = [(center[0], center[1] + 10), (center[0] - 10, center[1] - 5), (center[0] + 10, center[1] - 5)]
        elif self.gravity_dir == 1:  # Up
            points = [(center[0], center[1] - 10), (center[0] - 10, center[1] + 5), (center[0] + 10, center[1] + 5)]
        elif self.gravity_dir == 2:  # Left
            points = [(center[0] - 10, center[1]), (center[0] + 5, center[1] - 10), (center[0] + 5, center[1] + 10)]
        elif self.gravity_dir == 3:  # Right
            points = [(center[0] + 10, center[1]), (center[0] - 5, center[1] - 10), (center[0] - 5, center[1] + 10)]

        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    # --- Gymnasium Interface Compliance ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "checkpoints_collected": sum(1 for c in self.checkpoints if c['collected']),
            "shield_active": self.shield_active,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # To run, you need to `pip install pygame` and remove/comment the `os.environ` line at the top.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # We need a real display for manual play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gravity Maze")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print(f"Final Info: {info}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()