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
        "Controls: ←→ to move. ↑ or Space to jump. Collect coins and reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a jumping robot through a procedural obstacle course, collecting coins and reaching the exit before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG_TOP = (20, 30, 50)
        self.COLOR_BG_BOTTOM = (40, 60, 90)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_OBSTACLE = (100, 110, 120)
        self.COLOR_OBSTACLE_HL = (130, 140, 150)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0)

        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 1800  # 60 seconds * 30 fps
        self.EXIT_WORLD_X = 5000
        self.PLAYER_SCREEN_X = self.WIDTH // 3

        # Physics constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -14
        self.PLAYER_SPEED = 6
        self.MAX_FALL_SPEED = 15

        # Initial state variables (will be properly set in reset)
        self.player_rect = None
        self.player_vy = 0
        self.on_ground = False
        self.world_x = 0
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.game_outcome = ""
        self.obstacles = []
        self.coins = []
        self.particles = []
        self.stars = []
        self.last_obstacle_x = 0
        self.obstacle_gap = 300
        self.rng = None

        # Initialize state variables
        # Using a default seed for the first reset if none is provided
        self.reset(seed=42)
        # self.validate_implementation() # This is for debugging, can be commented out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use the provided seed for the random number generator
        if seed is not None:
            self.rng = random.Random(seed)
        # If no seed is provided and no rng exists, create a new one
        elif self.rng is None:
            # Using a fixed seed for the very first run if no seed is ever passed.
            # Subsequent resets without a seed will continue the sequence.
            self.rng = random.Random(random.randint(0, 1_000_000_000))

        # Player state
        self.player_rect = pygame.Rect(self.PLAYER_SCREEN_X, self.HEIGHT // 2, 24, 32)
        self.player_vy = 0
        self.on_ground = False

        # Game state
        self.world_x = 0
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.game_outcome = ""

        # World generation
        self.obstacles = []
        self.coins = []
        self.particles = []
        self.last_obstacle_x = 0
        self.obstacle_gap = 300

        # Create a starting platform
        start_platform = pygame.Rect(-self.WIDTH, self.HEIGHT - 50, self.WIDTH * 3, 50)
        self.obstacles.append(start_platform)

        # Procedurally generate the initial world
        while self.last_obstacle_x < self.world_x + self.WIDTH * 1.5:
            self._generate_chunk()

        # Parallax stars
        self.stars = [
            (self.rng.randint(0, self.WIDTH), self.rng.randint(0, self.HEIGHT), self.rng.uniform(0.2, 0.8))
            for _ in range(100)
        ]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused

        # --- Game Logic ---
        self._handle_input(movement, space_held)
        self._update_player_physics()
        self._handle_collisions()
        coin_reward = self._check_coin_collection()
        self._update_world()
        self._update_particles()

        self.steps += 1
        self.timer -= 1

        # --- Reward Calculation ---
        reward = 0.01 + coin_reward  # Survival + coin rewards

        # --- Termination Check ---
        terminated = False
        if self.player_rect.top > self.HEIGHT:
            terminated = True
            reward = -100.0
            self.game_outcome = "YOU FELL!"
        elif self.timer <= 0:
            terminated = True
            reward = 0  # No penalty, just end of time
            self.game_outcome = "TIME'S UP!"
        elif self.world_x + self.player_rect.centerx > self.EXIT_WORLD_X:
            terminated = True
            reward = 100.0
            self.score += int(max(0, self.timer) / self.FPS)  # Time bonus
            self.game_outcome = "YOU WIN!"
        elif self.steps >= self.MAX_STEPS * 2:  # Generous step limit
            terminated = True
            self.game_outcome = "EPISODE ENDED"

        self.game_over = terminated

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 3:  # Left
            self.world_x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.world_x += self.PLAYER_SPEED

        is_jump_action = (movement == 1 or space_held)
        if is_jump_action and self.on_ground:
            self.player_vy = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound()

    def _update_player_physics(self):
        self.player_vy += self.GRAVITY
        self.player_vy = min(self.player_vy, self.MAX_FALL_SPEED)
        self.player_rect.y += self.player_vy
        self.on_ground = False

    def _handle_collisions(self):
        player_moved_y = self.player_vy

        for obstacle in self.obstacles:
            screen_obstacle_rect = obstacle.move(-self.world_x, 0)
            if self.player_rect.colliderect(screen_obstacle_rect):
                if player_moved_y > 0 and self.player_rect.bottom - player_moved_y <= screen_obstacle_rect.top + 1:
                    self.player_rect.bottom = screen_obstacle_rect.top
                    self.player_vy = 0
                    self.on_ground = True
                elif player_moved_y < 0 and self.player_rect.top - player_moved_y >= screen_obstacle_rect.bottom - 1:
                    self.player_rect.top = screen_obstacle_rect.bottom
                    self.player_vy = 0.1
                else:
                    if self.player_rect.right > screen_obstacle_rect.left and self.player_rect.left < screen_obstacle_rect.left:
                        self.world_x -= self.PLAYER_SPEED
                    elif self.player_rect.left < screen_obstacle_rect.right and self.player_rect.right > screen_obstacle_rect.right:
                        self.world_x += self.PLAYER_SPEED

    def _check_coin_collection(self):
        collected_reward = 0
        for coin in self.coins[:]:
            screen_coin_rect = pygame.Rect(coin[0] - self.world_x, coin[1], 16, 16)
            if self.player_rect.colliderect(screen_coin_rect):
                self.coins.remove(coin)
                self.score += 10
                collected_reward += 1.0
                # sfx: coin_collect_sound()
                for _ in range(10):
                    self.particles.append([
                        screen_coin_rect.centerx, screen_coin_rect.centery,
                        self.rng.uniform(-3, 3), self.rng.uniform(-3, 3),
                        self.rng.randint(3, 6),
                        self.rng.randint(15, 25)
                    ])
        return collected_reward

    def _update_world(self):
        if self.last_obstacle_x < self.world_x + self.WIDTH * 1.5:
            self._generate_chunk()

        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_gap = max(150, self.obstacle_gap * 0.95)

        self.obstacles = [obs for obs in self.obstacles if obs.right - self.world_x > -self.WIDTH]
        self.coins = [coin for coin in self.coins if coin[0] - self.world_x > -50]

    def _generate_chunk(self):
        if self.last_obstacle_x > self.EXIT_WORLD_X - self.WIDTH:
            return

        gap = self.rng.randint(int(self.obstacle_gap * 0.8), int(self.obstacle_gap * 1.2))
        platform_x = self.last_obstacle_x + gap
        platform_y = self.rng.randint(self.HEIGHT // 2, self.HEIGHT - 80)
        platform_width = self.rng.randint(100, 300)
        platform_height = self.rng.randint(40, 100)

        new_platform = pygame.Rect(platform_x, platform_y, platform_width, platform_height)
        self.obstacles.append(new_platform)

        num_coins = self.rng.randint(1, 4)
        for i in range(num_coins):
            coin_x = platform_x + (platform_width // (num_coins + 1)) * (i + 1)
            coin_y = platform_y - 50
            self.coins.append((coin_x, coin_y))

        self.last_obstacle_x = new_platform.right

    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[5] -= 1
            if p[5] <= 0: self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_world_objects()
        self._render_player()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        grad_rect = pygame.Rect(0, 0, self.WIDTH, int(self.HEIGHT * 0.7))
        grad_surf = pygame.Surface(grad_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(grad_surf, (*self.COLOR_BG_TOP, 255), grad_surf.get_rect())
        pygame.draw.rect(grad_surf, (0, 0, 0, 0), (0, grad_rect.height - 1, self.WIDTH, 1))
        # This is a fast way to create a vertical gradient
        alpha_gradient = np.linspace(255, 0, grad_rect.height)[:, np.newaxis]
        alpha_surface = np.zeros((*grad_rect.size, 4), dtype=np.uint8)
        # FIX: Transpose the gradient to broadcast correctly.
        # Original alpha_gradient shape: (height, 1) -> (280, 1)
        # Target alpha_surface slice shape: (width, height) -> (640, 280)
        # Broadcasting (280, 1) to (640, 280) fails.
        # Transposing to (1, 280) allows broadcasting to (640, 280), creating a vertical gradient.
        alpha_surface[:, :, 3] = alpha_gradient.T
        alpha_surf = pygame.image.frombuffer(alpha_surface.tobytes(), grad_rect.size, 'RGBA')
        grad_surf.blit(alpha_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(grad_surf, (0, 0))

        for x, y, depth in self.stars:
            screen_x = (x - self.world_x * depth) % self.WIDTH
            color = int(200 * depth)
            pygame.gfxdraw.pixel(self.screen, int(screen_x), int(y), (color, color, color))

    def _render_world_objects(self):
        for obstacle in self.obstacles:
            screen_rect = obstacle.move(-self.world_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect, border_radius=3)
                highlight_rect = pygame.Rect(screen_rect.left, screen_rect.top, screen_rect.width, 5)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_HL, highlight_rect, border_top_left_radius=3,
                                 border_top_right_radius=3)

        for coin_x, coin_y in self.coins:
            screen_x = coin_x - self.world_x
            if -20 < screen_x < self.WIDTH + 20:
                spin_phase = (self.steps % self.FPS) / self.FPS
                width = 16 * abs(math.cos(spin_phase * 2 * math.pi))
                coin_rect = pygame.Rect(0, 0, max(2, width), 16)
                coin_rect.center = (int(screen_x), int(coin_y))
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, coin_rect)

        exit_screen_x = self.EXIT_WORLD_X - self.world_x
        if -50 < exit_screen_x < self.WIDTH + 50:
            exit_rect = pygame.Rect(exit_screen_x, 0, 40, self.HEIGHT)
            glow_alpha = 128 + 127 * math.sin(self.steps * 0.2)
            color = (self.COLOR_EXIT[0], self.COLOR_EXIT[1], self.COLOR_EXIT[2], glow_alpha)
            glow_surf = pygame.Surface((40, self.HEIGHT), pygame.SRCALPHA)
            glow_surf.fill(color)
            self.screen.blit(glow_surf, (int(exit_screen_x), 0), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, 3)

    def _render_player(self):
        squash = 4 if self.on_ground and self.player_vy == 0 else 0
        stretch = max(0, -self.player_vy * 0.5)
        p_rect = self.player_rect.copy()
        p_rect.height = max(8, p_rect.height + stretch - squash)
        p_rect.width = max(8, p_rect.width - stretch / 2 + squash)
        p_rect.y -= stretch
        p_rect.centerx = self.player_rect.centerx

        glow_rect = p_rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), glow_surf.get_rect())
        self.screen.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect, border_radius=4)

        eye_rect = pygame.Rect(0, 0, 6, 6)
        eye_rect.center = (p_rect.centerx + 5, p_rect.centery - 5)
        pygame.draw.ellipse(self.screen, (255, 255, 255), eye_rect)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, p[5] * 15))
            color = (*self.COLOR_COIN, alpha)
            surf = pygame.Surface((p[4] * 2, p[4] * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p[4], p[4]), p[4])
            self.screen.blit(surf, (p[0] - p[4], p[1] - p[4]), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        def draw_text(text, pos, font, color, shadow_color):
            text_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_shadow, (pos[0] + 2, pos[1] + 2))
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, pos)

        score_text = f"SCORE: {self.score}"
        time_text = f"TIME: {max(0, self.timer // self.FPS):02d}"

        draw_text(score_text, (10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_SHADOW)
        draw_text(time_text, (self.WIDTH - 150, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_SHADOW)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            text_surf = self.font_game_over.render(self.game_outcome, True, self.COLOR_TEXT)
            shadow_surf = self.font_game_over.render(self.game_outcome, True, self.COLOR_SHADOW)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(shadow_surf, text_rect.move(3, 3))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    env.reset(seed=42)

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Jumping Robot")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    obs, info = env.reset()

    while running:
        keys = pygame.key.get_pressed()
        movement = 0  # No-op
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()