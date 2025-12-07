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


# Helper class to allow adding custom attributes to a pygame.Rect
class PlatformRect(pygame.Rect):
    def __init__(self, *args, color, plat_type, move_speed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = color
        self.type = plat_type
        self.move_speed = move_speed


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to aim your jump (Up, Down, Left, Right). Hold Space for a power jump. "
        "You can only jump when on a platform."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms to reach the top. Collect stars for points. "
        "Green platforms are safe, blue are standard, and red platforms are moving and risky."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.MAX_STAGES = 3

        # Colors
        self.COLOR_BG_TOP = (40, 40, 80)
        self.COLOR_BG_BOTTOM = (100, 100, 160)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 255, 50)
        self.COLOR_PLATFORM_SAFE = (60, 200, 60)
        self.COLOR_PLATFORM_STANDARD = (60, 120, 220)
        self.COLOR_PLATFORM_RISKY = (220, 60, 60)
        self.COLOR_STAR = (255, 220, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        # Player Physics & Controls
        self.GRAVITY = 0.5
        self.PLAYER_SIZE = (12, 18)
        self.JUMP_Y_NORMAL = 10.0
        self.JUMP_Y_BOOST = 12.5
        self.JUMP_X = 6.0
        self.PLAYER_MAX_VEL_Y = 15.0

        # Game Mechanics
        self.PLATFORM_W_RANGE = (60, 120)
        self.PLATFORM_H = 15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_ui_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.np_random = None
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.on_ground_platform = None
        self.platforms = []
        self.stars = []
        self.particles = []
        self.score = 0
        self.stage = 1
        self.steps = 0
        self.game_over = False
        self.last_y_pos = 0

        # Initialize state variables
        # self.reset() is called by the wrapper, but we can call it here for standalone use
        # self.validate_implementation() # For testing during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.stage = 1
        self.game_over = False

        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 60]
        self.player_vel = [0, 0]
        self.last_y_pos = self.player_pos[1]

        self.platforms = []
        self.stars = []
        self.particles = []

        self._generate_stage()
        self.on_ground_platform = self.platforms[0]
        self.player_pos = [self.on_ground_platform.centerx, self.on_ground_platform.top - self.PLAYER_SIZE[1] / 2]


        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0

        # 1. Handle player actions (jumping)
        action_reward = self._handle_action(action)
        reward += action_reward

        # 2. Update physics and game objects
        self._update_player_physics()
        self._update_platforms()
        self._update_particles()

        # 3. Handle collisions and collect rewards
        collision_reward, stage_cleared = self._handle_collisions()
        reward += collision_reward

        # 4. Continuous height-based reward
        if self.player_pos[1] < self.last_y_pos:
            reward += 0.1
        elif self.player_pos[1] > self.last_y_pos:
            reward -= 0.1
        self.last_y_pos = self.player_pos[1]

        # 5. Check for stage clear
        if stage_cleared:
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                reward += 100  # Win game reward
                self.game_over = True
            else:
                reward += 100  # Stage clear reward
                # SFX: Stage Clear
                self._generate_stage()
                # Ensure player is on the new ground platform
                self.player_pos = [self.platforms[0].centerx, self.platforms[0].top - self.PLAYER_SIZE[1] / 2]
                self.player_vel = [0, 0]
                self.on_ground_platform = self.platforms[0]

        # 6. Check for termination conditions
        terminated = self.game_over or self._check_fall_termination()
        if self._check_fall_termination() and not self.game_over:
            reward = -100  # Fall penalty
            self.game_over = True
            terminated = True

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        if self.on_ground_platform is None:
            return 0

        movement = action[0]
        space_held = action[1] == 1
        # shift_held is not used in this game

        if movement == 0:  # No-op
            return 0

        # SFX: Jump
        self._create_particles(
            (self.player_pos[0], self.player_pos[1] + self.PLAYER_SIZE[1] / 2),
            10, self.on_ground_platform.color, (10, 20), (1, 3)
        )

        self.on_ground_platform = None
        jump_y_power = self.JUMP_Y_BOOST if space_held else self.JUMP_Y_NORMAL

        if movement == 1:  # Up
            self.player_vel = [0, -jump_y_power]
        elif movement == 2:  # Down (short hop)
            self.player_vel = [0, -jump_y_power * 0.4]
        elif movement == 3:  # Left
            self.player_vel = [-self.JUMP_X, -jump_y_power * 0.8]
        elif movement == 4:  # Right
            self.player_vel = [self.JUMP_X, -jump_y_power * 0.8]

        return 0

    def _update_player_physics(self):
        if self.on_ground_platform:
            self.player_vel = [0, 0]
            # Snap to platform surface
            self.player_pos[1] = self.on_ground_platform.top - self.PLAYER_SIZE[1] / 2
            # Move with platform
            if self.on_ground_platform.move_speed != 0:
                self.player_pos[0] += self.on_ground_platform.move_speed
                if self.player_pos[0] > self.WIDTH: self.player_pos[0] = 0
                if self.player_pos[0] < 0: self.player_pos[0] = self.WIDTH
        else:
            self.player_vel[1] += self.GRAVITY
            self.player_vel[1] = min(self.player_vel[1], self.PLAYER_MAX_VEL_Y)

            self.player_pos[0] += self.player_vel[0]
            self.player_pos[1] += self.player_vel[1]

            # Screen wrap horizontally
            if self.player_pos[0] > self.WIDTH: self.player_pos[0] = 0
            if self.player_pos[0] < 0: self.player_pos[0] = self.WIDTH

    def _update_platforms(self):
        for p in self.platforms:
            if p.move_speed != 0:
                p.x += p.move_speed
                if p.right < 0:
                    p.left = self.WIDTH
                if p.left > self.WIDTH:
                    p.right = 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        stage_cleared = False
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE[0] / 2,
            self.player_pos[1] - self.PLAYER_SIZE[1] / 2,
            self.PLAYER_SIZE[0], self.PLAYER_SIZE[1]
        )

        # Platform collisions (only if falling)
        if self.player_vel[1] > 0 and self.on_ground_platform is None:
            for p in self.platforms:
                if player_rect.colliderect(p) and player_rect.bottom < p.centery:
                    self.on_ground_platform = p
                    # SFX: Land
                    self._create_particles(
                        player_rect.midbottom, 15, p.color, (15, 25), (1, 4)
                    )
                    if p.type == 'risky':
                        reward -= 1
                    if p.type == 'goal':
                        stage_cleared = True
                    break

        # Star collisions
        collided_stars = []
        for star in self.stars:
            star_rect = pygame.Rect(star['pos'][0] - 8, star['pos'][1] - 8, 16, 16)
            if player_rect.colliderect(star_rect):
                collided_stars.append(star)
                reward += 10
                self.score += 100
                # SFX: Star collect
                self._create_particles(star['pos'], 20, self.COLOR_STAR, (20, 30), (2, 5))
        self.stars = [s for s in self.stars if s not in collided_stars]

        return reward, stage_cleared

    def _check_fall_termination(self):
        return self.player_pos[1] > self.HEIGHT + self.PLAYER_SIZE[1]

    def _generate_stage(self):
        self.platforms.clear()
        self.stars.clear()

        # Base platform
        base_plat = PlatformRect(
            0, self.HEIGHT - 30, self.WIDTH, 30,
            color=self.COLOR_PLATFORM_SAFE,
            plat_type='safe',
            move_speed=0
        )
        self.platforms.append(base_plat)

        # Procedural platforms
        current_y = self.HEIGHT - 120
        last_x = self.WIDTH / 2

        while current_y > 80:
            width = self.np_random.integers(self.PLATFORM_W_RANGE[0], self.PLATFORM_W_RANGE[1])
            x_offset = self.np_random.uniform(-160, 160)
            y_offset = self.np_random.uniform(80, 130)

            x = last_x + x_offset
            x = np.clip(x, width / 2, self.WIDTH - width / 2)

            plat_type_roll = self.np_random.random()
            max_move_speed = min(2.0, self.stage * 0.5)

            if self.stage > 1 and plat_type_roll < 0.2 + self.stage * 0.1:
                color = self.COLOR_PLATFORM_RISKY
                plat_type = 'risky'
                move_speed = self.np_random.uniform(0.5, max_move_speed) * (1 if self.np_random.random() > 0.5 else -1)
            else:
                color = self.COLOR_PLATFORM_STANDARD
                plat_type = 'standard'
                move_speed = 0

            plat = PlatformRect(
                x - width / 2, current_y, width, self.PLATFORM_H,
                color=color,
                plat_type=plat_type,
                move_speed=move_speed
            )
            self.platforms.append(plat)

            # Add a star sometimes
            if self.np_random.random() < 0.5:
                star_pos = [plat.centerx + self.np_random.uniform(-30, 30),
                            plat.top - self.np_random.uniform(30, 60)]
                self.stars.append({'pos': star_pos})

            last_x = x
            current_y -= y_offset

        # Goal platform
        goal_plat = PlatformRect(
            self.WIDTH / 2 - 50, 40, 100, 20,
            color=self.COLOR_PLATFORM_SAFE,
            plat_type='goal',
            move_speed=0
        )
        self.platforms.append(goal_plat)

    def _create_particles(self, pos, count, color, life_range, speed_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_range[0], speed_range[1])
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(life_range[0], life_range[1]),
                'color': color
            })

    def _get_observation(self):
        # 1. Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # 2. Render game elements
        for star in self.stars:
            pos = (int(star['pos'][0]), int(star['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_STAR)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_STAR)

        for p in self.platforms:
            pygame.draw.rect(self.screen, p.color, p, border_radius=3)

        for p in self.particles:
            size = max(0, int(p['life'] * 0.2))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (*pos, size, size))

        # Draw player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        w, h = self.PLAYER_SIZE
        points = [(px, py - h / 2), (px - w / 2, py + h / 2), (px + w / 2, py + h / 2)]
        glow_points = [(px, py - h / 2 - 2), (px - w / 2 - 2, py + h / 2 + 2), (px + w / 2 + 2, py + h / 2 + 2)]
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # 3. Render UI overlay
        self._render_text(f"SCORE: {self.score}", (10, 10), self.font_ui)
        self._render_text(f"STAGE: {self.stage}/{self.MAX_STAGES}", (self.WIDTH - 10, 10), self.font_ui, "topright")

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, pos, font, anchor="topleft"):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        setattr(text_rect, anchor, pos)

        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hopper")
    clock = pygame.time.Clock()

    running = True
    terminated = False
    truncated = False

    while running:
        if terminated or truncated:
            print(f"Game Over! Final Score: {env.score}")
            obs, info = env.reset()
            terminated = False
            truncated = False

        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0  # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(env.FPS)

    env.close()