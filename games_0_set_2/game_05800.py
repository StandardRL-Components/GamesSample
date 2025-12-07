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
        "Controls: Use ← and → to move, and ↑ to jump. Collect all the crystals!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore procedurally generated Crystal Caverns, collecting crystals while avoiding deadly pitfalls."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = self.SCREEN_WIDTH * 3

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_BG_ACCENT = (25, 30, 50)
        self.COLOR_PLAYER = (0, 192, 255)
        self.COLOR_PLATFORM = (60, 65, 85)
        self.COLOR_PIT = (0, 0, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.CRYSTAL_COLORS = [
            (255, 0, 255), (0, 255, 255), (255, 255, 0),
            (255, 128, 0), (128, 0, 255), (0, 255, 128)
        ]

        # Game constants
        self.NUM_CRYSTALS = 20
        self.NUM_PITS = 3
        self.MAX_STEPS = 2000

        # Physics constants
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -9
        self.PLAYER_ACCEL = 0.6
        self.PLAYER_FRICTION = 0.85
        self.PLAYER_MAX_SPEED = 6

        # Initialize state variables
        self.player_rect = None
        self.player_vel = None
        self.on_ground = False
        self.platforms = []
        self.pits = []
        self.crystals = []
        self.particles = []
        self.camera_x = 0
        self.rng = None

        # This will be called once to set up the initial state, including RNG
        self.reset()
        # self.validate_implementation() # Optional validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_level()

        self.player_rect = pygame.Rect(100, 200, 20, 20)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False

        self.particles = []
        self.camera_x = 0

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.pits = []
        self.crystals = []

        # Start with an initial platform
        current_x = 0
        y = self.SCREEN_HEIGHT - 50
        w = 300
        self.platforms.append(pygame.Rect(current_x, y, w, 50))
        current_x += w

        pit_count = 0

        while current_x < self.WORLD_WIDTH - self.SCREEN_WIDTH:
            # Decide to make a platform or a pit
            is_pit = self.rng.random() < 0.2 and pit_count < self.NUM_PITS

            if is_pit:
                gap_w = self.rng.integers(120, 200)
                self.pits.append(pygame.Rect(current_x, 0, gap_w, self.SCREEN_HEIGHT * 2))
                current_x += gap_w
                pit_count += 1
            else:
                gap_w = self.rng.integers(30, 100)
                current_x += gap_w

                y = self.rng.integers(
                    max(150, y - 80),
                    min(self.SCREEN_HEIGHT - 50, y + 80)
                )
                w = self.rng.integers(150, 400)

                self.platforms.append(pygame.Rect(current_x, y, w, self.SCREEN_HEIGHT - y))
                current_x += w

        # Add a final platform to ensure the world is fully traversable
        if current_x < self.WORLD_WIDTH:
             self.platforms.append(pygame.Rect(current_x, y, self.WORLD_WIDTH - current_x, self.SCREEN_HEIGHT - y))

        # Place crystals
        for _ in range(self.NUM_CRYSTALS):
            # FIX: Select platform by index to avoid numpy object conversion
            platform_idx = self.rng.integers(len(self.platforms))
            platform = self.platforms[platform_idx]

            # Ensure crystal placement is valid
            if platform.width < 40: continue

            crystal_x = self.rng.integers(platform.left + 20, platform.right - 20)
            crystal_y = platform.top - self.rng.integers(20, 150)
            
            # Use Python's random for choosing from a list of tuples
            color = self.CRYSTAL_COLORS[self.rng.integers(len(self.CRYSTAL_COLORS))]

            self.crystals.append({
                "rect": pygame.Rect(crystal_x, crystal_y, 12, 12),
                "color": color,
                "angle": self.rng.random() * 2 * math.pi,
                "bob_offset": self.rng.random() * 5
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right

        # --- Player Input and Movement ---
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL

        self.player_vel.x *= self.PLAYER_FRICTION
        self.player_vel.x = max(-self.PLAYER_MAX_SPEED, min(self.PLAYER_MAX_SPEED, self.player_vel.x))
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0

        if movement == 1 and self.on_ground:  # Jump
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump

        # --- Physics and Collisions ---
        self.player_vel.y += self.GRAVITY
        self.player_rect.x += int(self.player_vel.x)

        # Horizontal collision
        for platform in self.platforms:
            if self.player_rect.colliderect(platform):
                if self.player_vel.x > 0:
                    self.player_rect.right = platform.left
                    self.player_vel.x = 0
                elif self.player_vel.x < 0:
                    self.player_rect.left = platform.right
                    self.player_vel.x = 0

        self.player_rect.y += int(self.player_vel.y)
        self.on_ground = False

        # Vertical collision
        for platform in self.platforms:
            if self.player_rect.colliderect(platform):
                if self.player_vel.y > 0:
                    self.player_rect.bottom = platform.top
                    self.player_vel.y = 0
                    self.on_ground = True
                elif self.player_vel.y < 0:
                    self.player_rect.top = platform.bottom
                    self.player_vel.y = 0

        # Boundary checks
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.WORLD_WIDTH, self.player_rect.right)

        # --- Update game state ---
        self.steps += 1
        reward = -0.01  # Time penalty
        if movement == 0:
            reward -= 0.1 # Penalty for inaction

        # Crystal collection
        collected_indices = []
        for i, crystal in enumerate(self.crystals):
            if self.player_rect.colliderect(crystal["rect"]):
                collected_indices.append(i)
                reward += 10
                self.score += 10
                self._spawn_particles(crystal["rect"].center, crystal["color"])
                # sfx: crystal_collect

        # Remove collected crystals
        for i in sorted(collected_indices, reverse=True):
            del self.crystals[i]

        # --- Termination checks ---
        terminated = False
        truncated = False
        # 1. Fell in a pit
        if self.player_rect.top > self.SCREEN_HEIGHT:
            terminated = True
            reward = -10
            self.game_over = True
            # sfx: fall_death

        # 2. Collected all crystals
        if not self.crystals:
            terminated = True
            reward += 100
            self.score += 100
            self.game_over = True
            # sfx: win_level

        # 3. Max steps reached
        if self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limits
            self.game_over = True

        self._update_particles()
        
        # If the game ends, terminated and truncated can't both be true
        if terminated and truncated:
            truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "color": color,
                "life": self.rng.integers(20, 40)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # Update camera to follow player
        if self.player_rect:
            self.camera_x += (self.player_rect.centerx - self.camera_x - self.SCREEN_WIDTH / 2) * 0.1
            self.camera_x = max(0, min(self.WORLD_WIDTH - self.SCREEN_WIDTH, self.camera_x))

        # --- Drawing ---
        self.screen.fill(self.COLOR_BG)

        # Background accents
        for i in range(10):
            x = (i * 200 - self.camera_x * 0.5) % (self.SCREEN_WIDTH + 400) - 200
            pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (x, 0, 100, self.SCREEN_HEIGHT))

        # Draw platforms
        for platform in self.platforms:
            view_rect = platform.move(-self.camera_x, 0)
            if view_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, view_rect)

        # Draw crystals
        for crystal in self.crystals:
            crystal["angle"] += 0.05
            bob = math.sin(crystal["angle"] * 2 + crystal["bob_offset"]) * 3
            cx, cy = crystal["rect"].centerx, crystal["rect"].centery + bob
            view_pos = (cx - self.camera_x, cy)

            # Glow effect
            glow_radius = 15
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*crystal["color"], 60), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (view_pos[0] - glow_radius, view_pos[1] - glow_radius))

            # Crystal shape
            size = 6
            points = [
                (view_pos[0], view_pos[1] - size),
                (view_pos[0] + size, view_pos[1]),
                (view_pos[0], view_pos[1] + size),
                (view_pos[0] - size, view_pos[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, crystal["color"])
            pygame.gfxdraw.filled_polygon(self.screen, points, crystal["color"])

        # Draw player
        if self.player_rect:
            view_player_rect = self.player_rect.move(-self.camera_x, 0)
            
            # Player glow
            player_center = view_player_rect.center
            glow_radius = 25
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (player_center[0] - glow_radius, player_center[1] - glow_radius))

            pygame.draw.rect(self.screen, self.COLOR_PLAYER, view_player_rect)

        # Draw particles
        for p in self.particles:
            view_pos = (p["pos"][0] - self.camera_x, p["pos"][1])
            alpha = max(0, min(255, int(255 * (p["life"] / 30))))
            color = (*p["color"], alpha)
            radius = int(p["life"] / 8)
            if radius > 0:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (view_pos[0] - radius, view_pos[1] - radius))

        # Draw UI
        self._render_text(f"Score: {self.score}", (15, 10), self.font_small)
        crystal_text = f"Crystals: {len(self.crystals)} left"
        text_width = self.font_small.size(crystal_text)[0]
        self._render_text(crystal_text, (self.SCREEN_WIDTH - text_width - 15, 10), self.font_small)

        if self.game_over:
            msg = "LEVEL COMPLETE!" if not self.crystals else "GAME OVER"
            text_width, text_height = self.font_large.size(msg)
            pos = (self.SCREEN_WIDTH/2 - text_width/2, self.SCREEN_HEIGHT/2 - text_height/2)
            self._render_text(msg, pos, self.font_large)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, pos, font):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_remaining": len(self.crystals),
            "player_pos": (self.player_rect.x, self.player_rect.y) if self.player_rect else (0,0),
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset(seed=42)

    # Setup Pygame window for human play
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_UP]:
            movement = 1

        # The action space is MultiDiscrete, so we form the action array
        action = [movement, 0, 0] # Space and Shift are not used in this game

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()